"""
SEAM: Surprise-gated Episodic Adaptive Memory

A biologically-inspired dual memory system for neural networks.
"""

from .config import MemoryConfig, ExperimentConfig
from .buffer import DecayingBuffer
from .store import SurpriseGatedStore
from .layer import DualMemoryLayer
from .embedder import SimpleEmbedder

__all__ = [
    'MemoryConfig',
    'ExperimentConfig',
    'DecayingBuffer',
    'SurpriseGatedStore',
    'DualMemoryLayer',
    'SimpleEmbedder',
]

__version__ = '0.1.0'
