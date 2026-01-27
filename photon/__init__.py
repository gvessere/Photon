"""
PHOTON: Hierarchical Latent Language Model

Paper-aligned defaults (PHOTON-600M / Table 6):
- Multi-level latent hierarchy (tokens -> L1 -> L2)
- Top-down autoregressive generation (optional latent AR head disabled by default)
- Converters sized to paper params (1664 → 1664 with R=4, d_int=832)
- RoPE positional encoding per level
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
