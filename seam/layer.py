"""
SEAM Dual Memory Layer - Combined buffer and store
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from .config import MemoryConfig
from .buffer import DecayingBuffer
from .store import SurpriseGatedStore


class DualMemoryLayer(nn.Module):
    """Combined memory layer with both buffer and store."""

    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config

        self.buffer = DecayingBuffer(config)
        self.store = SurpriseGatedStore(config)

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

        output = x + memory_contribution

        return output, info

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
