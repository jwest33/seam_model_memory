"""
SEAM Decaying Buffer - Episodic memory with temporal decay
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any

from .config import MemoryConfig


class DecayingBuffer(nn.Module):
    """
    Type 1: Episodic buffer that retains all history with temporal decay.
    Analogous to hippocampal short-term memory / working memory.
    """

    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config

        # Memory banks
        self.register_buffer('keys', torch.zeros(config.buffer_slots, config.hidden_dim))
        self.register_buffer('values', torch.zeros(config.buffer_slots, config.hidden_dim))
        self.register_buffer('activation', torch.zeros(config.buffer_slots))
        self.register_buffer('write_count', torch.zeros(config.buffer_slots))
        self.register_buffer('cycle', torch.tensor(0))
        self.register_buffer('content_tags', torch.zeros(config.buffer_slots, dtype=torch.long))

        # Projections for addressing
        self.key_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.value_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.query_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

    def decay_step(self, num_steps: int = 1):
        """Apply temporal decay to all memory activations."""
        decay_factor = self.config.decay_rate ** num_steps
        self.activation.mul_(decay_factor)
        self.cycle.add_(num_steps)

    def write(
        self,
        x: torch.Tensor,
        write_mask: Optional[torch.Tensor] = None,
        content_tag: int = 0
    ) -> Dict[str, Any]:
        """
        Write to buffer with content-addressable addressing.

        Args:
            x: [batch, seq_len, hidden_dim]
            write_mask: [batch, seq_len] - which positions to write
            content_tag: Integer tag for tracking content type

        Returns:
            Dictionary with write statistics
        """
        batch, seq_len, d = x.shape
        stats = {'novel_writes': 0, 'reinforcements': 0, 'slots_used': []}

        keys = self.key_proj(x)
        values = self.value_proj(x)

        # Compute similarity to existing memory keys
        similarity = torch.matmul(keys, self.keys.T) / math.sqrt(d)
        max_sim, best_match = similarity.max(dim=-1)

        is_novel = max_sim < self.config.novelty_threshold

        for b in range(batch):
            for s in range(seq_len):
                if write_mask is not None and not write_mask[b, s]:
                    continue

                if is_novel[b, s]:
                    slot = self.activation.argmin().item()
                    alpha = self.config.write_alpha_novel
                    stats['novel_writes'] += 1
                else:
                    slot = best_match[b, s].item()
                    alpha = self.config.write_alpha_reinforce
                    stats['reinforcements'] += 1

                stats['slots_used'].append(slot)

                self.keys[slot] = (1 - alpha) * self.keys[slot] + alpha * keys[b, s]
                self.values[slot] = (1 - alpha) * self.values[slot] + alpha * values[b, s]
                self.activation[slot] = min(1.0, self.activation[slot] + self.config.activation_boost)
                self.write_count[slot] += 1
                self.content_tags[slot] = content_tag

        return stats

    def read(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Content-addressable read from buffer.

        Returns:
            retrieved: [batch, seq_len, hidden_dim]
            attention_weights: [batch, seq_len, buffer_slots]
        """
        queries = self.query_proj(x)

        attn_logits = torch.matmul(queries, self.keys.T) / math.sqrt(self.config.hidden_dim)

        inactive_mask = self.activation < 0.01
        attn_logits = attn_logits.masked_fill(
            inactive_mask.unsqueeze(0).unsqueeze(0),
            float('-inf')
        )

        # Weight by activation level
        activation_weights = torch.log(self.activation.clamp(min=1e-8))
        attn_logits = attn_logits + activation_weights

        attn_weights = F.softmax(attn_logits / self.config.temperature, dim=-1)
        retrieved = torch.matmul(attn_weights, self.values)

        return retrieved, attn_weights

    def get_state(self) -> Dict[str, Any]:
        """Get current buffer state for analysis."""
        active_mask = self.activation > 0.01
        return {
            'num_active_slots': active_mask.sum().item(),
            'total_slots': self.config.buffer_slots,
            'mean_activation': self.activation[active_mask].mean().item() if active_mask.any() else 0,
            'max_activation': self.activation.max().item(),
            'cycle': self.cycle.item(),
            'active_indices': active_mask.nonzero(as_tuple=True)[0].tolist(),
            'activations': self.activation[active_mask].tolist() if active_mask.any() else [],
            'content_tags': self.content_tags[active_mask].tolist() if active_mask.any() else []
        }

    def clear(self):
        """Clear all buffer memory."""
        self.keys.zero_()
        self.values.zero_()
        self.activation.zero_()
        self.write_count.zero_()
        self.content_tags.zero_()
