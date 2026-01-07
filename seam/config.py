"""
SEAM Configuration Classes
"""

import torch
from dataclasses import dataclass, field
from typing import List


@dataclass
class MemoryConfig:
    """Configuration for dual memory system."""
    hidden_dim: int = 768
    buffer_slots: int = 256
    store_slots: int = 128
    decay_rate: float = 0.95
    # Adaptive threshold parameters
    z_score_threshold: float = 1.0      # Write if z-score > 1.0 (~top 16%)
    ema_alpha: float = 0.1              # EMA smoothing factor for running stats
    capacity_pressure: float = 0.5      # How much store fullness raises threshold
    temperature: float = 1.0
    novelty_threshold: float = 0.5
    write_alpha_novel: float = 0.3
    write_alpha_reinforce: float = 0.1
    activation_boost: float = 0.5
    consolidation_similarity_threshold: float = 0.8


@dataclass
class ExperimentConfig:
    """Configuration for the name recall experiment."""
    # Model settings
    model_name: str = "D:\\models\\gemma-3-1b-it"  # Default to small model for testing
    memory_layer_positions: List[int] = field(default_factory=lambda: [3, 6, 9])
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Experiment settings
    initial_name: str = "Alice"
    corrected_name: str = "Bob"
    num_reinforcement_turns: int = 5
    num_decay_cycles_between_convos: int = 10

    # Output settings
    output_dir: str = "./experiment_results"
    save_memory_snapshots: bool = True
    verbose: bool = True
