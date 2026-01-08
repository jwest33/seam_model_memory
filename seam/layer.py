"""
SEAM Dual Memory Layer - Combined buffer and store
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from .config import MemoryConfig
from .buffer import DecayingBuffer
from .store import SurpriseGatedStore
from .attention import MemoryCrossAttentionLayer


class DualMemoryLayer(nn.Module):
    """
    Combined memory layer with both buffer and store.

    Supports two read modes:
    - 'additive': Original approach - gated addition to residual (doesn't work well)
    - 'cross_attention': Cross-attention to memory slots (principled retrieval)
    """

    def __init__(self, config: MemoryConfig, read_mode: str = 'cross_attention'):
        super().__init__()
        self.config = config
        self.read_mode = read_mode

        self.buffer = DecayingBuffer(config)
        self.store = SurpriseGatedStore(config)

        if read_mode == 'cross_attention':
            # Cross-attention for memory retrieval
            self.cross_attn = MemoryCrossAttentionLayer(
                hidden_dim=config.hidden_dim,
                num_heads=8,
                dropout=0.0
            )
        else:
            # Legacy additive read
            self.read_gate = nn.Sequential(
                nn.Linear(config.hidden_dim * 3, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, 2),
                nn.Sigmoid()
            )
            self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
            nn.init.normal_(self.out_proj.weight, std=0.01)
            nn.init.zeros_(self.out_proj.bias)
            self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        update_memory: bool = True,
        content_tag: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through dual memory layer.

        Returns:
            output: [batch, seq_len, hidden_dim]
            info: Dictionary with memory statistics
        """
        info = {}

        # Surprise computation
        surprise, prediction, surprise_stats = self.store.compute_surprise(x, context)
        info['surprise'] = surprise
        info['surprise_stats'] = surprise_stats

        # Memory writes
        if update_memory:
            self.buffer.decay_step()
            buffer_write_stats = self.buffer.write(x, content_tag=content_tag)
            store_write_stats = self.store.write_surprising(x, surprise, content_tag=content_tag)
            info['buffer_write_stats'] = buffer_write_stats
            info['store_write_stats'] = store_write_stats

        # Memory reads - different modes
        if self.read_mode == 'cross_attention':
            output, cross_attn_info = self._read_cross_attention(x)
            info['cross_attention'] = cross_attn_info
        else:
            output = self._read_additive(x, info)

        return output, info

    def _read_cross_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Read from memory using cross-attention."""
        # Combine buffer and store memories for cross-attention
        # Use buffer keys/values and store keys/values

        # Get active memory from buffer
        buffer_mask = self.buffer.activation > 0
        buffer_keys = self.buffer.keys  # [buffer_slots, hidden_dim]
        buffer_values = self.buffer.values

        # Get active memory from store
        store_mask = self.store.surprise_level > 0
        store_keys = self.store.keys  # [store_slots, hidden_dim]
        store_values = self.store.values

        # Concatenate buffer and store memories
        all_keys = torch.cat([buffer_keys, store_keys], dim=0)
        all_values = torch.cat([buffer_values, store_values], dim=0)
        all_mask = torch.cat([buffer_mask, store_mask], dim=0)

        # Cross-attend to combined memory
        output, attn_info = self.cross_attn(
            x,
            all_keys,
            all_values,
            all_mask
        )

        attn_info['buffer_slots_active'] = buffer_mask.sum().item()
        attn_info['store_slots_active'] = store_mask.sum().item()

        return output, attn_info

    def _read_additive(self, x: torch.Tensor, info: Dict[str, Any]) -> torch.Tensor:
        """Legacy additive read mode."""
        # Memory reads
        buffer_retrieved, buffer_attn = self.buffer.read(x)
        store_retrieved, store_attn = self.store.read(x)

        info['buffer_attention'] = buffer_attn
        info['store_attention'] = store_attn

        # Gated combination
        gate_input = torch.cat([x, buffer_retrieved, store_retrieved], dim=-1)
        gates = self.read_gate(gate_input)

        buffer_gate = gates[..., 0:1]
        store_gate = gates[..., 1:2]

        info['buffer_gate'] = buffer_gate.squeeze(-1)
        info['store_gate'] = store_gate.squeeze(-1)

        memory_content = buffer_gate * buffer_retrieved + store_gate * store_retrieved
        memory_contribution = self.out_proj(self.norm(memory_content))

        return x + memory_contribution

    def get_state(self) -> Dict[str, Any]:
        """Get combined state of both memory systems."""
        return {
            'buffer': self.buffer.get_state(),
            'store': self.store.get_state()
        }

    def clear(self):
        """Clear all memory."""
        self.buffer.clear()
        self.store.clear()
