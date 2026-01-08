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
    model_name: str = "./gemma-3-1b-it-null-space-abliterated"  # Default to abliterated model
    memory_layer_positions: List[int] = field(default_factory=lambda: [3, 6, 9])
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Memory architecture settings (must match checkpoint if loading)
    hidden_dim: int = 768
    buffer_slots: int = 256
    store_slots: int = 128

    # Experiment settings
    initial_name: str = "Alice"
    corrected_name: str = "Sarah"
    num_reinforcement_turns: int = 5
    num_decay_cycles_between_convos: int = 10

    # Output settings
    output_dir: str = "./experiment_results"
    save_memory_snapshots: bool = True
    verbose: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training memory layers."""
    # Optimizer settings
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Training settings
    num_epochs: int = 10
    batch_size: int = 1  # Memory-intensive, keep small
    gradient_accumulation_steps: int = 4
    eval_every: int = 50
    save_every: int = 200

    # Data settings
    num_train_names: int = 500
    num_val_names: int = 50

    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./training_logs"

    # Model settings (inherit from ExperimentConfig or override)
    model_name: str = "./gemma-3-1b-it-null-space-abliterated"
    memory_layer_positions: List[int] = field(default_factory=lambda: [3, 6, 9])
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_dim: int = 768
    buffer_slots: int = 256
    store_slots: int = 128
