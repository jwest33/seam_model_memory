"""
SEAM Surprise-Gated Store - Consolidation based on free energy principle
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any

from .config import MemoryConfig


class SurpriseGatedStore(nn.Module):
    """
    Type 2: Surprise-based consolidation store.
    Implements free energy principle - writes occur when prediction error is high.
    """

    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config

        # Memory banks
        self.register_buffer('keys', torch.zeros(config.store_slots, config.hidden_dim))
        self.register_buffer('values', torch.zeros(config.store_slots, config.hidden_dim))
        self.register_buffer('surprise_level', torch.zeros(config.store_slots))
        self.register_buffer('access_count', torch.zeros(config.store_slots))
        self.register_buffer('slot_age', torch.zeros(config.store_slots))
        self.register_buffer('content_tags', torch.zeros(config.store_slots, dtype=torch.long))

        # Running statistics for adaptive threshold
        self.register_buffer('surprise_mean', torch.tensor(0.5))
        self.register_buffer('surprise_std', torch.tensor(0.2))
        self.register_buffer('num_observations', torch.tensor(0))

        # Projections
        self.key_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.value_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.query_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        # Surprise estimation network
        self.predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        self.context_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

    def _compute_adaptive_threshold(self) -> float:
        """Compute dynamic threshold based on running stats and capacity."""
        # Base threshold from z-score
        base = self.surprise_mean + self.config.z_score_threshold * self.surprise_std

        # Capacity pressure: raise threshold when store is fuller
        utilization = (self.surprise_level > 0).float().mean()
        pressure = 1.0 + utilization * self.config.capacity_pressure

        return (base * pressure).item()

    def compute_surprise(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Compute free energy / surprise for each position.

        Returns:
            surprise: [batch, seq_len] - scalar surprise per position
            prediction: [batch, seq_len, hidden_dim]
            stats: Dictionary with surprise statistics
        """
        if context is None:
            context = F.pad(x[:, :-1, :], (0, 0, 1, 0), value=0)

        context_repr = self.context_proj(context)
        prediction = self.predictor(context_repr)

        # Surprise as L2 distance (simplified free energy)
        mse = ((x - prediction) ** 2).mean(dim=-1)
        surprise = torch.sqrt(mse)

        # Update running statistics with EMA
        batch_mean = surprise.mean()
        batch_std = surprise.std().clamp(min=1e-6)

        if self.num_observations == 0:
            self.surprise_mean.copy_(batch_mean)
            self.surprise_std.copy_(batch_std)
        else:
            alpha = self.config.ema_alpha
            self.surprise_mean.mul_(1 - alpha).add_(batch_mean * alpha)
            self.surprise_std.mul_(1 - alpha).add_(batch_std * alpha)
        self.num_observations.add_(1)

        adaptive_threshold = self._compute_adaptive_threshold()

        stats = {
            'mean_surprise': surprise.mean().item(),
            'max_surprise': surprise.max().item(),
            'min_surprise': surprise.min().item(),
            'std_surprise': surprise.std().item(),
            'adaptive_threshold': adaptive_threshold,
            'above_threshold': (surprise > adaptive_threshold).sum().item()
        }

        return surprise, prediction, stats

    def write_surprising(
        self,
        x: torch.Tensor,
        surprise: torch.Tensor,
        content_tag: int = 0,
        min_surprise: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Write only positions that exceed surprise threshold.

        Returns:
            Dictionary with write statistics
        """
        threshold = min_surprise if min_surprise is not None else self._compute_adaptive_threshold()
        batch, seq_len, d = x.shape
        stats = {'writes': 0, 'updates': 0, 'slots_used': [], 'surprise_values': []}

        keys = self.key_proj(x)
        values = self.value_proj(x)

        surprising_mask = surprise > threshold

        if not surprising_mask.any():
            return stats

        for b in range(batch):
            surprising_indices = surprising_mask[b].nonzero(as_tuple=True)[0]

            for idx in surprising_indices:
                surprise_val = surprise[b, idx].item()
                key = keys[b, idx]
                value = values[b, idx]

                similarity = F.cosine_similarity(
                    key.unsqueeze(0),
                    self.keys,
                    dim=-1
                )

                importance = (
                    self.surprise_level * 0.5 +
                    torch.log1p(self.access_count) * 0.3 -
                    self.slot_age * 0.001
                )

                if similarity.max() > self.config.consolidation_similarity_threshold:
                    slot = similarity.argmax().item()
                    if surprise_val > self.surprise_level[slot]:
                        alpha = 0.3
                        self.keys[slot] = (1 - alpha) * self.keys[slot] + alpha * key
                        self.values[slot] = (1 - alpha) * self.values[slot] + alpha * value
                        self.surprise_level[slot] = surprise_val
                        stats['updates'] += 1
                else:
                    slot = importance.argmin().item()
                    self.keys[slot] = key
                    self.values[slot] = value
                    self.surprise_level[slot] = surprise_val
                    self.access_count[slot] = 0
                    stats['writes'] += 1

                self.content_tags[slot] = content_tag
                self.slot_age[slot] = 0
                stats['slots_used'].append(slot)
                stats['surprise_values'].append(surprise_val)

        self.slot_age.add_(1)
        return stats

    def read(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read from store with attention."""
        queries = self.query_proj(x)

        attn_logits = torch.matmul(queries, self.keys.T) / math.sqrt(self.config.hidden_dim)

        empty_mask = self.surprise_level == 0
        attn_logits = attn_logits.masked_fill(
            empty_mask.unsqueeze(0).unsqueeze(0),
            float('-inf')
        )

        attn_weights = F.softmax(attn_logits, dim=-1)

        access_contribution = attn_weights.sum(dim=(0, 1))
        self.access_count.add_(access_contribution)

        retrieved = torch.matmul(attn_weights, self.values)

        return retrieved, attn_weights

    def get_state(self) -> Dict[str, Any]:
        """Get current store state for analysis."""
        active_mask = self.surprise_level > 0
        return {
            'num_active_slots': active_mask.sum().item(),
            'total_slots': self.config.store_slots,
            'mean_surprise': self.surprise_level[active_mask].mean().item() if active_mask.any() else 0,
            'max_surprise': self.surprise_level.max().item(),
            'active_indices': active_mask.nonzero(as_tuple=True)[0].tolist(),
            'surprise_levels': self.surprise_level[active_mask].tolist() if active_mask.any() else [],
            'access_counts': self.access_count[active_mask].tolist() if active_mask.any() else [],
            'content_tags': self.content_tags[active_mask].tolist() if active_mask.any() else []
        }

    def clear(self):
        """Clear all store memory."""
        self.keys.zero_()
        self.values.zero_()
        self.surprise_level.zero_()
        self.access_count.zero_()
        self.slot_age.zero_()
        self.content_tags.zero_()
