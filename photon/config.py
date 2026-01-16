"""
PHOTON Configuration

Dimensions based on Table 7 from the paper:
- Token embedding: d_embed_enc = 480
- Level-1 latent: d_latent = 1920 (= 4 * 480 from concat chunker)
- Converter internal: d_converter = 2432 (Table 7 shows 9728 -> 2432)
- The 9728 = 4 * 2432 suggests 4 latents concatenated as input
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PhotonConfig:
    # Vocabulary
    vocab_size: int = 32000
    
    # Chunking factors
    C1: int = 4   # tokens per chunk at level 1
    C2: int = 4   # level-1 units per chunk at level 2
    
    # Embedding dimensions (Table 7 aligned)
    d_embed_enc: int = 480          # Token embedding dim for encoder
    d_latent: int = 1920            # = C1 * d_embed_enc, level-1/2 latent dim
    d_converter: int = 2432         # Internal converter dim from Table 7
    
    # Conditioning prefix lengths (R_l)
    R2: int = 4   # Number of conditioning tokens for level-2 decoder
    R1: int = 4   # Number of conditioning tokens for level-1 (token) decoder
    
    # Transformer hyperparams (defaults sized for 2×T4, ~350M params)
    n_heads: int = 8
    d_ff: int = 2048        # FFN hidden dimension
    n_layers_enc: int = 4   # Encoder transformer layers per level
    n_layers_dec: int = 4   # Decoder transformer layers per level
    n_layers_latent_ar: int = 2  # Latent AR head layers
    
    # RoPE settings
    rope_theta: float = 10000.0
    rope_dim: Optional[int] = None  # If None, use d_latent // n_heads
    
    # Loss weighting (Paper Eq. 7: L = L_LM + λ_ctx * L_ctx + λ_rec * L_rec)
    lambda_lm: float = 1.0      # Weight for token prediction loss
    lambda_ctx: float = 1.0     # Weight for next-context prediction (L2 AR)
    lambda_rec: float = 1.0     # Weight for reconstruction loss (L2→L1 prediction)
    
    # Training settings
    gradient_checkpointing: bool = False
    use_sdpa: bool = True  # Use scaled_dot_product_attention
    
    # Detach conditioning paths to prevent encoder-decoder collusion
    # Set to False to allow gradients to flow from decoder back to encoder
    # (closer to paper's "reconstruction" but risks mode collapse)
    detach_conditioning: bool = True
    
    # EOS token id (set during data loading)
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    
    def __post_init__(self):
        # Validate chunk divisibility
        assert self.d_latent == self.C1 * self.d_embed_enc, \
            f"d_latent ({self.d_latent}) must equal C1 * d_embed_enc ({self.C1 * self.d_embed_enc})"
        
        # Set rope_dim if not specified
        if self.rope_dim is None:
            self.rope_dim = self.d_latent // self.n_heads
    
    @property
    def block_size(self) -> int:
        """Minimum sequence length divisible by all chunk factors."""
        return self.C1 * self.C2
    
    @property
    def d_head(self) -> int:
        """Dimension per attention head."""
        return self.d_latent // self.n_heads
