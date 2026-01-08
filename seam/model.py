"""
SEAM Memory-Augmented Model - LLM with integrated memory layers
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

from .config import MemoryConfig, ExperimentConfig
from .layer import DualMemoryLayer

logger = logging.getLogger(__name__)


class MemoryAugmentedModel(nn.Module):
    """
    Wraps a causal LLM and injects DualMemoryLayer at specified positions.

    Memory layers are inserted after transformer layers at the positions
    specified in config.memory_layer_positions (e.g., [3, 6, 9]).
    """

    def __init__(
        self,
        config: ExperimentConfig,
        memory_config: Optional[MemoryConfig] = None
    ):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # Load base model and tokenizer
        logger.info(f"Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=config.device
        )

        # Get hidden dimension from model config
        self.hidden_dim = self.base_model.config.hidden_size
        logger.info(f"Model hidden dimension: {self.hidden_dim}")

        # Create memory config with correct hidden dim
        if memory_config is None:
            memory_config = MemoryConfig(
                hidden_dim=self.hidden_dim,
                buffer_slots=config.buffer_slots,
                store_slots=config.store_slots,
            )
        self.memory_config = memory_config

        # Create memory layers for each injection position
        self.memory_layer_positions = config.memory_layer_positions
        self.memory_layers = nn.ModuleDict({
            str(pos): DualMemoryLayer(memory_config).to(self.device)
            for pos in self.memory_layer_positions
        })

        logger.info(f"Memory layers at positions: {self.memory_layer_positions}")

        # Skip weight initialization - will be trained instead
        # self._init_memory_from_model_weights()

        # State for tracking memory operations during forward
        self._current_memory_info: Dict[int, Dict[str, Any]] = {}
        self._update_memory = True
        self._content_tag = 0

        # Text-based memory store (stores actual text for retrieval)
        # This enables prompt-injection based recall
        self._text_memory: List[Dict[str, Any]] = []

        # Register hooks on transformer layers
        self._register_hooks()

    def _get_transformer_layers(self) -> nn.ModuleList:
        """Get the list of transformer layers from the base model."""
        # Handle different model architectures
        if hasattr(self.base_model, 'model'):
            model = self.base_model.model
        else:
            model = self.base_model

        # Try common attribute names for transformer layers
        for attr in ['layers', 'h', 'blocks', 'decoder', 'encoder']:
            if hasattr(model, attr):
                layers = getattr(model, attr)
                if hasattr(layers, 'layers'):  # For decoder.layers
                    layers = layers.layers
                if isinstance(layers, nn.ModuleList):
                    return layers

        raise ValueError(f"Could not find transformer layers in model architecture")

    def _init_memory_from_model_weights(self):
        """Initialize memory cross-attention from model's own attention weights."""
        layers = self._get_transformer_layers()

        for pos in self.memory_layer_positions:
            if pos >= len(layers):
                continue

            layer = layers[pos]
            memory_layer = self.memory_layers[str(pos)]

            # Skip if not using cross-attention mode
            if memory_layer.read_mode != 'cross_attention':
                continue

            cross_attn = memory_layer.cross_attn.cross_attn

            # Find attention module in the layer
            # Try common attribute names for different architectures
            attn = None
            for attr in ['self_attn', 'attention', 'attn']:
                if hasattr(layer, attr):
                    attn = getattr(layer, attr)
                    break

            if attn is None:
                logger.warning(f"Could not find attention in layer {pos}")
                continue

            # Copy Q, K, V projections from model attention to memory cross-attention
            # Handle different projection naming conventions
            try:
                # Try to find Q, K, V projections
                q_proj = k_proj = v_proj = o_proj = None

                # Gemma/Llama style: q_proj, k_proj, v_proj, o_proj
                if hasattr(attn, 'q_proj'):
                    q_proj = attn.q_proj
                    k_proj = attn.k_proj
                    v_proj = attn.v_proj
                    o_proj = attn.o_proj if hasattr(attn, 'o_proj') else None

                # GPT-2 style: c_attn (combined QKV)
                elif hasattr(attn, 'c_attn'):
                    # Skip - would need to split combined projection
                    logger.info(f"Layer {pos} uses combined QKV - using default init")
                    continue

                # Falcon style: query_key_value
                elif hasattr(attn, 'query_key_value'):
                    logger.info(f"Layer {pos} uses combined QKV - using default init")
                    continue

                if q_proj is not None:
                    # Copy weights (handle dimension mismatches by using subset)
                    with torch.no_grad():
                        # Q projection for queries (from hidden states)
                        q_weight = q_proj.weight.data
                        if q_weight.shape == cross_attn.q_proj.weight.shape:
                            cross_attn.q_proj.weight.copy_(q_weight)
                            if q_proj.bias is not None and cross_attn.q_proj.bias is not None:
                                cross_attn.q_proj.bias.copy_(q_proj.bias.data)
                        else:
                            # Handle GQA (grouped query attention) - take first num_heads
                            out_features = cross_attn.q_proj.weight.shape[0]
                            cross_attn.q_proj.weight.copy_(q_weight[:out_features])
                            if q_proj.bias is not None and cross_attn.q_proj.bias is not None:
                                cross_attn.q_proj.bias.copy_(q_proj.bias.data[:out_features])

                        # K projection for memory keys
                        k_weight = k_proj.weight.data
                        if k_weight.shape == cross_attn.k_proj.weight.shape:
                            cross_attn.k_proj.weight.copy_(k_weight)
                            if k_proj.bias is not None and cross_attn.k_proj.bias is not None:
                                cross_attn.k_proj.bias.copy_(k_proj.bias.data)
                        else:
                            # Handle KV with fewer heads
                            out_features = cross_attn.k_proj.weight.shape[0]
                            in_features = cross_attn.k_proj.weight.shape[1]
                            # Repeat K weights to match expected dimensions
                            k_expanded = k_weight.repeat(out_features // k_weight.shape[0] + 1, 1)[:out_features, :in_features]
                            cross_attn.k_proj.weight.copy_(k_expanded)

                        # V projection for memory values
                        v_weight = v_proj.weight.data
                        if v_weight.shape == cross_attn.v_proj.weight.shape:
                            cross_attn.v_proj.weight.copy_(v_weight)
                            if v_proj.bias is not None and cross_attn.v_proj.bias is not None:
                                cross_attn.v_proj.bias.copy_(v_proj.bias.data)
                        else:
                            out_features = cross_attn.v_proj.weight.shape[0]
                            in_features = cross_attn.v_proj.weight.shape[1]
                            v_expanded = v_weight.repeat(out_features // v_weight.shape[0] + 1, 1)[:out_features, :in_features]
                            cross_attn.v_proj.weight.copy_(v_expanded)

                        # Output projection
                        if o_proj is not None:
                            o_weight = o_proj.weight.data
                            if o_weight.shape == cross_attn.out_proj.weight.shape:
                                cross_attn.out_proj.weight.copy_(o_weight * 0.1)  # Scale down
                                if o_proj.bias is not None and cross_attn.out_proj.bias is not None:
                                    cross_attn.out_proj.bias.copy_(o_proj.bias.data * 0.1)

                    logger.info(f"Initialized memory cross-attention at layer {pos} from model weights")

            except Exception as e:
                logger.warning(f"Could not copy attention weights for layer {pos}: {e}")
                continue

    def _register_hooks(self):
        """Register forward hooks on transformer layers to inject memory."""
        layers = self._get_transformer_layers()
        num_layers = len(layers)
        logger.info(f"Model has {num_layers} transformer layers")

        self._hooks = []

        for pos in self.memory_layer_positions:
            if pos >= num_layers:
                logger.warning(f"Memory position {pos} exceeds model layers ({num_layers}), skipping")
                continue

            def make_hook(layer_pos):
                def hook(module, input, output):
                    # output is typically (hidden_states, ...) or just hidden_states
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                        rest = output[1:]
                    else:
                        hidden_states = output
                        rest = None

                    # Apply memory layer
                    memory_layer = self.memory_layers[str(layer_pos)]
                    augmented, info = memory_layer(
                        hidden_states.float(),  # Memory ops in fp32
                        update_memory=self._update_memory,
                        content_tag=self._content_tag
                    )

                    # Store info for later retrieval
                    self._current_memory_info[layer_pos] = info

                    # Return with original dtype
                    augmented = augmented.to(hidden_states.dtype)

                    if rest is not None:
                        return (augmented,) + rest
                    return augmented

                return hook

            hook_handle = layers[pos].register_forward_hook(make_hook(pos))
            self._hooks.append(hook_handle)
            logger.info(f"Registered memory hook at layer {pos}")

    def set_memory_mode(self, update_memory: bool = True, content_tag: int = 0):
        """Set whether to update memory during forward pass."""
        self._update_memory = update_memory
        self._content_tag = content_tag

    def get_last_memory_info(self) -> Dict[int, Dict[str, Any]]:
        """Get memory info from the last forward pass."""
        return self._current_memory_info.copy()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Any:
        """Forward pass through memory-augmented model."""
        self._current_memory_info = {}
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        update_memory: bool = True,
        content_tag: int = 0,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate text with memory augmentation.

        Returns:
            generated_text: The generated response
            memory_info: Memory statistics from generation
        """
        self.set_memory_mode(update_memory=update_memory, content_tag=content_tag)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        outputs = self.base_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs
        )

        # Decode only the new tokens
        input_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[0, input_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text, self.get_last_memory_info()

    def get_memory_state(self) -> Dict[str, Dict[str, Any]]:
        """Get state of all memory layers."""
        return {
            f"layer_{pos}": self.memory_layers[str(pos)].get_state()
            for pos in self.memory_layer_positions
        }

    def clear_memory(self):
        """Clear all memory layers."""
        for layer in self.memory_layers.values():
            layer.clear()

    def apply_decay(self, num_cycles: int = 1):
        """Apply decay to all memory layers."""
        for layer in self.memory_layers.values():
            layer.buffer.decay_step(num_cycles)

    def save_memory(self, path: str):
        """Save memory state to disk."""
        state = {
            'memory_layers': {
                pos: layer.state_dict()
                for pos, layer in self.memory_layers.items()
            },
            'config': {
                'memory_layer_positions': self.memory_layer_positions,
                'hidden_dim': self.memory_config.hidden_dim,
                'buffer_slots': self.memory_config.buffer_slots,
                'store_slots': self.memory_config.store_slots,
            }
        }
        torch.save(state, path)
        logger.info(f"Memory saved to: {path}")

    def load_memory(self, path: str):
        """Load memory state from disk."""
        state = torch.load(path, map_location=self.device, weights_only=False)

        # Validate config
        saved_config = state['config']
        if saved_config['memory_layer_positions'] != self.memory_layer_positions:
            raise ValueError(
                f"Memory position mismatch: saved={saved_config['memory_layer_positions']}, "
                f"current={self.memory_layer_positions}"
            )

        # Load state dicts
        for pos, state_dict in state['memory_layers'].items():
            self.memory_layers[pos].load_state_dict(state_dict)

        logger.info(f"Memory loaded from: {path}")

    def load_checkpoint(self, path: str):
        """Load trained memory layer weights from a training checkpoint."""
        state = torch.load(path, map_location=self.device, weights_only=False)

        # Validate config
        saved_config = state.get('config', {})
        if saved_config.get('memory_layer_positions') != self.memory_layer_positions:
            logger.warning(
                f"Memory position mismatch: saved={saved_config.get('memory_layer_positions')}, "
                f"current={self.memory_layer_positions}. Attempting to load anyway."
            )

        # Load memory layer state dicts (includes trained weights)
        for pos, state_dict in state['memory_layers'].items():
            if pos in self.memory_layers:
                self.memory_layers[pos].load_state_dict(state_dict)
                logger.info(f"Loaded checkpoint weights for memory layer {pos}")
            else:
                logger.warning(f"Checkpoint has layer {pos} but model doesn't, skipping")

        logger.info(f"Checkpoint loaded from: {path}")
