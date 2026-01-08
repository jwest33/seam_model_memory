"""
SEAM Memory Cross-Attention

Cross-attention mechanism for explicit memory retrieval.
Model hidden states query memory slots via attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any


class MemoryCrossAttention(nn.Module):
    """
    Cross-attention layer for memory retrieval.

    Queries: Model hidden states [batch, seq_len, hidden_dim]
    Keys/Values: Memory slots [num_slots, hidden_dim]

    This allows the model to explicitly attend to and retrieve from memory.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Query projection (from model hidden states)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        # Key/Value projections (from memory slots)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Gate to control memory influence (learnable)
        self.memory_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Layer norm for memory input
        self.memory_norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights so memory doesn't dominate initially."""
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Cross-attend to memory.

        Args:
            hidden_states: [batch, seq_len, hidden_dim] - model hidden states
            memory_keys: [num_slots, hidden_dim] - memory slot keys
            memory_values: [num_slots, hidden_dim] - memory slot values
            memory_mask: [num_slots] - True for valid slots, False for empty

        Returns:
            output: [batch, seq_len, hidden_dim] - memory-augmented hidden states
            info: Attention statistics
        """
        batch_size, seq_len, _ = hidden_states.shape
        num_slots = memory_keys.shape[0]

        # Check if memory is empty
        if memory_mask is None or not memory_mask.any():
            # No memory to attend to - return zeros
            return torch.zeros_like(hidden_states), {
                'attended': False,
                'num_valid_slots': 0
            }

        # Normalize memory
        memory_keys = self.memory_norm(memory_keys)
        memory_values = self.memory_norm(memory_values)

        # Project queries from hidden states
        queries = self.q_proj(hidden_states)  # [batch, seq_len, hidden_dim]

        # Project keys/values from memory (shared across batch)
        keys = self.k_proj(memory_keys)  # [num_slots, hidden_dim]
        values = self.v_proj(memory_values)  # [num_slots, hidden_dim]

        # Reshape for multi-head attention
        # queries: [batch, num_heads, seq_len, head_dim]
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # keys/values: [1, num_heads, num_slots, head_dim] (broadcast over batch)
        keys = keys.view(1, num_slots, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(1, num_slots, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_weights = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        # [batch, num_heads, seq_len, num_slots]

        # Apply memory mask (mask out empty slots)
        if memory_mask is not None:
            # memory_mask: [num_slots] -> [1, 1, 1, num_slots]
            mask = memory_mask.view(1, 1, 1, -1)
            attn_weights = attn_weights.masked_fill(~mask, float('-inf'))

        # Softmax over memory slots
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Handle NaN from all-masked softmax
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)

        # Attend to values
        context = torch.matmul(attn_probs, values)
        # [batch, num_heads, seq_len, head_dim]

        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        # Output projection
        memory_output = self.out_proj(context)

        # Compute gate based on hidden states and memory output
        gate_input = torch.cat([hidden_states, memory_output], dim=-1)
        gate = self.memory_gate(gate_input)  # [batch, seq_len, 1]

        # Apply gate
        output = gate * memory_output

        # Compute attention stats
        info = {
            'attended': True,
            'num_valid_slots': memory_mask.sum().item() if memory_mask is not None else num_slots,
            'mean_attention': attn_probs.mean().item(),
            'max_attention': attn_probs.max().item(),
            'gate_mean': gate.mean().item(),
        }

        return output, info


class MemoryCrossAttentionLayer(nn.Module):
    """
    Full cross-attention layer with residual connection and layer norm.
    Drop-in replacement for memory read in DualMemoryLayer.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        self.cross_attn = MemoryCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_values: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply cross-attention to memory with residual connection.

        Returns:
            output: hidden_states + memory_contribution
            info: Attention statistics
        """
        # Cross-attend to memory
        memory_contribution, info = self.cross_attn(
            self.norm(hidden_states),
            memory_keys,
            memory_values,
            memory_mask
        )

        # Residual connection
        output = hidden_states + memory_contribution

        return output, info
