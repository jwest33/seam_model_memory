"""
SEAM: Surprise-gated Episodic Adaptive Memory

A biologically-inspired dual memory system for neural networks.
"""

from .config import MemoryConfig, ExperimentConfig, TrainingConfig
from .buffer import DecayingBuffer
from .store import SurpriseGatedStore
from .layer import DualMemoryLayer
from .embedder import SimpleEmbedder
from .model import MemoryAugmentedModel
from .attention import MemoryCrossAttention, MemoryCrossAttentionLayer

__all__ = [
    'MemoryConfig',
    'ExperimentConfig',
    'TrainingConfig',
    'DecayingBuffer',
    'SurpriseGatedStore',
    'DualMemoryLayer',
    'SimpleEmbedder',
    'MemoryAugmentedModel',
    'MemoryCrossAttention',
    'MemoryCrossAttentionLayer',
]

__version__ = '0.1.0'
