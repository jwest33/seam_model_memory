"""
SEAM Simple Embedder - Character-level text encoding for testing
"""

import torch
import torch.nn as nn


class SimpleEmbedder(nn.Module):
    """
    Simple embedding layer for testing without full LLM.
    Uses character-level encoding with learned embeddings.
    """

    def __init__(self, hidden_dim: int = 768, vocab_size: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(512, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, text: str) -> torch.Tensor:
        """Convert text to embeddings."""
        device = self.embed.weight.device

        # Character-level encoding
        char_ids = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long, device=device)
        char_ids = char_ids.unsqueeze(0)  # [1, seq_len]

        positions = torch.arange(len(text), device=device).unsqueeze(0)

        embeddings = self.embed(char_ids) + self.pos_embed(positions)
        embeddings = self.proj(embeddings)

        return embeddings  # [1, seq_len, hidden_dim]
