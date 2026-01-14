"""
Baseline Transformer (GPT-style decoder-only)

A baseline implementation for comparison with PHOTON.
"""

from .model import BaselineConfig, BaselineLM
from .train import train_baseline

__all__ = ["BaselineConfig", "BaselineLM", "train_baseline"]
