"""
PHOTON: Hierarchical Latent Language Model

A faithful implementation of PHOTON with:
- Multi-level latent hierarchy (tokens -> L1 -> L2)
- Top-down autoregressive generation
- Table 7 matched converter dimensions
- RoPE positional encoding per level
- Gaussian NLL latent loss
- DeepSpeed ZeRO-3 / Accelerate compatible
"""

from .config import PhotonConfig
from .model import PhotonLM
from .data import create_dataloaders, collate_fn
from .inference import generate_photon

__all__ = [
    "PhotonConfig",
    "PhotonLM", 
    "create_dataloaders",
    "collate_fn",
    "generate_photon",
]
