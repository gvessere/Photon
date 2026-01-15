"""
PHOTON Model Architecture

Complete implementation with:
- SDPA (scaled_dot_product_attention) for efficient attention
- RoPE positional encoding at all levels
- Table 7 matched converter (1D conv)
- Gaussian NLL latent loss
- Top-level latent AR head
- Gradient checkpointing support
"""

import math
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import PhotonConfig


# =============================================================================
# Rotary Position Embedding (RoPE)
# =============================================================================

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding for hierarchical positions.
    
    Supports three position types:
    - Token position within chunk
    - Chunk index within stream  
    - Level position (for cross-level attention)
    """
    
    def __init__(self, dim: int, theta: float = 10000.0, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.max_seq_len = max_seq_len
        
        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache for cos/sin
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
    
    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = max(seq_len, self.max_seq_len)
            t = torch.arange(self._seq_len_cached, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)
    
    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply rotary embedding to input tensor.
        
        Args:
            x: [B, H, T, D] or [B, T, H, D]
            position_ids: [B, T] position indices, or None for sequential
        
        Returns:
            Rotary-embedded tensor of same shape
        """
        seq_len = x.shape[-2] if x.dim() == 4 else x.shape[1]
        self._update_cache(seq_len, x.device, x.dtype)
        
        if position_ids is None:
            cos = self._cos_cached[:seq_len]
            sin = self._sin_cached[:seq_len]
        else:
            cos = self._cos_cached[position_ids]
            sin = self._sin_cached[position_ids]
        
        return self._apply_rotary(x, cos, sin)
    
    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding: x * cos + rotate_half(x) * sin"""
        # x: [B, H, T, D] -> split into two halves
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat([-x2, x1], dim=-1)
        
        # Expand cos/sin to match x shape
        if cos.dim() == 2:  # [T, D]
            cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, D]
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif cos.dim() == 3:  # [B, T, D]
            cos = cos.unsqueeze(1)  # [B, 1, T, D]
            sin = sin.unsqueeze(1)
        
        return x * cos + rotated * sin


# =============================================================================
# Attention with SDPA
# =============================================================================

class CausalSelfAttention(nn.Module):
    """
    Causal self-attention with SDPA and RoPE support.
    
    Uses F.scaled_dot_product_attention for efficiency (FlashAttention backend).
    """
    
    def __init__(self, d_model: int, n_heads: int, rope: Optional[RotaryEmbedding] = None,
                 use_sdpa: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_sdpa = use_sdpa
        
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.rope = rope
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                is_causal: bool = True, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            attn_mask: Optional attention mask (additive, -inf for masked)
            is_causal: If True, apply causal mask
            position_ids: Optional position indices for RoPE
        """
        B, T, D = x.shape
        
        # QKV projection
        qkv = self.qkv(x)  # [B, T, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for attention: [B, H, T, D_head]
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # Apply RoPE if available
        if self.rope is not None:
            q = self.rope(q, position_ids)
            k = self.rope(k, position_ids)
        
        # Attention computation
        if self.use_sdpa:
            # Use PyTorch's optimized SDPA (FlashAttention backend)
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                is_causal=is_causal and attn_mask is None,
                dropout_p=0.0,
            )
        else:
            # Manual attention (fallback)
            scale = 1.0 / math.sqrt(self.d_head)
            att = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            if is_causal:
                causal = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
                att = att + causal
            
            if attn_mask is not None:
                att = att + attn_mask
            
            att = F.softmax(att, dim=-1)
            y = torch.matmul(att, v)
        
        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(y)


# =============================================================================
# Feed-Forward Network (MLP)
# =============================================================================

class MLP(nn.Module):
    """SwiGLU-style MLP."""
    
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_hidden, bias=False)
        self.up = nn.Linear(d_model, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, d_model, bias=False)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.gate(x)) * self.up(x))


# =============================================================================
# Transformer Block
# =============================================================================

class TransformerBlock(nn.Module):
    """Pre-norm transformer block with RoPE attention."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 rope: Optional[RotaryEmbedding] = None, use_sdpa: bool = True):
        super().__init__()
        self.ln1 = nn.RMSNorm(d_model)
        self.ln2 = nn.RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, rope=rope, use_sdpa=use_sdpa)
        self.mlp = MLP(d_model, d_ff)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                is_causal: bool = True, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask, is_causal=is_causal, position_ids=position_ids)
        x = x + self.mlp(self.ln2(x))
        return x


# =============================================================================
# Context Transformer (Encoder/Decoder stack)
# =============================================================================

class CtxTransformer(nn.Module):
    """
    Stack of transformer blocks for encoding/decoding at each level.
    
    Supports:
    - Causal, bidirectional, or block-causal masking
    - Gradient checkpointing
    - RoPE positioning
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int,
                 rope_theta: float = 10000.0, use_sdpa: bool = True,
                 gradient_checkpointing: bool = False):
        super().__init__()
        
        self.rope = RotaryEmbedding(d_model // n_heads, theta=rope_theta)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, rope=self.rope, use_sdpa=use_sdpa)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.RMSNorm(d_model)
        self.gradient_checkpointing = gradient_checkpointing
    
    def forward(self, x: torch.Tensor, is_causal: bool = True,
                attn_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
            is_causal: Apply causal masking
            attn_mask: Optional custom attention mask
            position_ids: Optional position indices for RoPE
        """
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(
                    blk, x, attn_mask, is_causal, position_ids,
                    use_reentrant=False
                )
            else:
                x = blk(x, attn_mask=attn_mask, is_causal=is_causal, position_ids=position_ids)
        
        return self.ln_f(x)


# =============================================================================
# Chunkers (Bottom-up encoding)
# =============================================================================

class ConcatChunker(nn.Module):
    """
    Concatenation chunker for level 1: groups tokens and concatenates features.
    
    [B, T, D] -> [B, T/block, block*D]
    """
    
    def __init__(self, block: int):
        super().__init__()
        self.block = block
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        assert T % self.block == 0, f"T ({T}) must be divisible by block ({self.block})"
        return x.view(B, T // self.block, self.block * D)


class LinearChunker(nn.Module):
    """
    Linear chunker for level 2+: concatenate then project.
    
    [B, T, D_in] -> [B, T/block, D_out]
    """
    
    def __init__(self, block: int, d_in: int, d_out: int):
        super().__init__()
        self.block = block
        self.proj = nn.Linear(block * d_in, d_out, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        assert T % self.block == 0, f"T ({T}) must be divisible by block ({self.block})"
        x = x.view(B, T // self.block, self.block * D)
        return self.proj(x)


# =============================================================================
# Table 7 Matched Converter (1D Conv based)
# =============================================================================

class TableMatchedConverter(nn.Module):
    """
    Converter U_theta matching Table 7 dimensions.
    
    Table 7 shows: 9728 -> 2432
    - 9728 = 4 * 2432 suggests concatenating 4 latents (or R previous latents)
    - Uses 1D convolution for expansion to R conditioning vectors
    
    Input: [B, d_in] single latent
    Output: [B, R, d_out] conditioning prefix
    """
    
    def __init__(self, d_in: int, d_out: int, R: int, d_internal: int = 2432):
        super().__init__()
        self.R = R
        self.d_out = d_out
        self.d_internal = d_internal
        
        # First project to internal dimension
        self.proj_in = nn.Linear(d_in, d_internal, bias=False)
        
        # 1D conv to expand: treats the latent as a 1-length sequence
        # Kernel operates on features, expands to R positions
        self.conv = nn.Conv1d(
            in_channels=d_internal,
            out_channels=R * d_internal,
            kernel_size=1,
            groups=1
        )
        
        # Project back to output dimension
        self.proj_out = nn.Linear(d_internal, d_out, bias=False)
        
        # Layer norm for stability
        self.ln = nn.RMSNorm(d_internal)
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: [B, d_in] or [B, N, d_in]
        
        Returns:
            [B, R, d_out] or [B, N, R, d_out]
        """
        has_seq_dim = latent.dim() == 3
        
        if has_seq_dim:
            B, N, D = latent.shape
            latent = latent.reshape(B * N, D)
        else:
            B = latent.size(0)
            N = 1
        
        # Project to internal dim
        x = self.proj_in(latent)  # [B*N, d_internal]
        x = self.ln(x)
        
        # 1D conv expansion: [B*N, d_internal, 1] -> [B*N, R*d_internal, 1]
        x = x.unsqueeze(-1)  # [B*N, d_internal, 1]
        x = self.conv(x)     # [B*N, R*d_internal, 1]
        x = x.squeeze(-1)    # [B*N, R*d_internal]
        
        # Reshape and project
        x = x.view(-1, self.R, self.d_internal)  # [B*N, R, d_internal]
        x = self.proj_out(x)  # [B*N, R, d_out]
        
        if has_seq_dim:
            x = x.view(B, N, self.R, self.d_out)
        
        return x


# =============================================================================
# Latent AR Head (Top-level autoregressive)
# =============================================================================

class LatentARHead(nn.Module):
    """
    Autoregressive head for top-level latent generation.
    
    Generates X_hat_g^(L) from X_hat_{g-1}^(L) without re-encoding.
    This enables the "lightspeed" property of PHOTON.
    
    Uses a small causal transformer over the latent stream.
    """
    
    def __init__(self, d_latent: int, n_heads: int, d_ff: int, n_layers: int,
                 rope_theta: float = 10000.0, use_sdpa: bool = True):
        super().__init__()
        
        self.d_latent = d_latent
        
        # Transformer for latent sequence modeling
        self.transformer = CtxTransformer(
            d_model=d_latent,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            rope_theta=rope_theta,
            use_sdpa=use_sdpa,
            gradient_checkpointing=False  # Small model, no need
        )
        
        # Prediction heads for Gaussian parameterization
        self.mean_head = nn.Linear(d_latent, d_latent, bias=False)
        self.logvar_head = nn.Linear(d_latent, d_latent, bias=False)
        
        # Initialize logvar to predict small variance initially
        nn.init.zeros_(self.logvar_head.weight)
    
    def forward(self, latent_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent_seq: [B, T, d_latent] sequence of latents
        
        Returns:
            mean: [B, T, d_latent] predicted mean for next latent
            logvar: [B, T, d_latent] predicted log variance
        """
        h = self.transformer(latent_seq, is_causal=True)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        return mean, logvar
    
    def sample(self, prev_latent: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Sample next latent given previous latent.
        
        Args:
            prev_latent: [B, d_latent] or [B, T, d_latent]
        
        Returns:
            [B, d_latent] sampled next latent
        """
        if prev_latent.dim() == 2:
            prev_latent = prev_latent.unsqueeze(1)
        
        mean, logvar = self.forward(prev_latent)
        mean = mean[:, -1, :]      # [B, d_latent]
        logvar = logvar[:, -1, :]  # [B, d_latent]
        
        if temperature == 0:
            return mean
        
        std = torch.exp(0.5 * logvar) * temperature
        return mean + std * torch.randn_like(std)


# =============================================================================
# Gaussian NLL Loss for Latents
# =============================================================================

class GaussianLatentLoss(nn.Module):
    """
    Gaussian negative log-likelihood loss for latent prediction.
    
    NLL = 0.5 * (log_var + (pred - target)^2 / exp(log_var))
    
    With learned variance, this is more expressive than MSE.
    """
    
    def __init__(self, min_logvar: float = -10.0, max_logvar: float = 10.0):
        super().__init__()
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
    
    def forward(self, pred_mean: torch.Tensor, pred_logvar: torch.Tensor,
                target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred_mean: [B, ...] predicted mean
            pred_logvar: [B, ...] predicted log variance
            target: [B, ...] target values
            mask: Optional [B, ...] mask (1 = valid, 0 = ignore)
        
        Returns:
            Scalar loss
        """
        # Clamp logvar for stability
        logvar = torch.clamp(pred_logvar, self.min_logvar, self.max_logvar)
        
        # Gaussian NLL
        var = torch.exp(logvar)
        nll = 0.5 * (logvar + (pred_mean - target) ** 2 / var)
        
        if mask is not None:
            nll = nll * mask
            loss = nll.sum() / mask.sum().clamp(min=1)
        else:
            loss = nll.mean()
        
        # Clamp to non-negative to prevent gaming with extreme logvar
        return loss.clamp(min=0.0)


# =============================================================================
# PHOTON Language Model
# =============================================================================

class PhotonLM(nn.Module):
    """
    PHOTON: Hierarchical Latent Language Model
    
    Complete implementation with:
    - Two-level latent hierarchy (tokens -> L1 -> L2)
    - Top-level latent AR generation
    - Table 7 matched converters
    - RoPE at all levels
    - Gaussian NLL latent loss
    - Gradient checkpointing support
    """
    
    def __init__(self, cfg: PhotonConfig):
        super().__init__()
        self.cfg = cfg
        
        # --- Encoder side (bottom-up) ---
        self.enc_embed = nn.Embedding(cfg.vocab_size, cfg.d_embed_enc)
        
        # Level 1: tokens -> latents
        self.enc_chunk1 = ConcatChunker(block=cfg.C1)
        self.enc_ctx1 = CtxTransformer(
            d_model=cfg.d_latent,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            n_layers=cfg.n_layers_enc,
            rope_theta=cfg.rope_theta,
            use_sdpa=cfg.use_sdpa,
            gradient_checkpointing=cfg.gradient_checkpointing
        )
        
        # Level 2: L1 latents -> L2 latents
        self.enc_chunk2 = LinearChunker(block=cfg.C2, d_in=cfg.d_latent, d_out=cfg.d_latent)
        self.enc_ctx2 = CtxTransformer(
            d_model=cfg.d_latent,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            n_layers=cfg.n_layers_enc,
            rope_theta=cfg.rope_theta,
            use_sdpa=cfg.use_sdpa,
            gradient_checkpointing=cfg.gradient_checkpointing
        )
        
        # --- Top-level latent AR (for generation without re-encoding) ---
        self.latent_ar_head = LatentARHead(
            d_latent=cfg.d_latent,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            n_layers=cfg.n_layers_latent_ar,
            rope_theta=cfg.rope_theta,
            use_sdpa=cfg.use_sdpa
        )
        
        # --- Decoder side (top-down) ---
        # Learned start latents
        self.start_latent_l2 = nn.Parameter(torch.randn(cfg.d_latent) * 0.02)
        self.start_latent_l1 = nn.Parameter(torch.randn(cfg.d_latent) * 0.02)
        
        # Level 2 -> Level 1 decoder
        self.dec_conv2 = TableMatchedConverter(
            d_in=cfg.d_latent,
            d_out=cfg.d_latent,
            R=cfg.R2,
            d_internal=cfg.d_converter
        )
        self.dec_ctx2 = CtxTransformer(
            d_model=cfg.d_latent,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            n_layers=cfg.n_layers_dec,
            rope_theta=cfg.rope_theta,
            use_sdpa=cfg.use_sdpa,
            gradient_checkpointing=cfg.gradient_checkpointing
        )
        
        # Latent prediction heads (for Gaussian loss)
        self.latent_mean_head = nn.Linear(cfg.d_latent, cfg.d_latent, bias=False)
        self.latent_logvar_head = nn.Linear(cfg.d_latent, cfg.d_latent, bias=False)
        nn.init.zeros_(self.latent_logvar_head.weight)
        
        # Level 1 -> Token decoder
        self.dec_conv1 = TableMatchedConverter(
            d_in=cfg.d_latent,
            d_out=cfg.d_latent,
            R=cfg.R1,
            d_internal=cfg.d_converter
        )
        self.dec_embed = nn.Embedding(cfg.vocab_size, cfg.d_latent)
        self.dec_ctx1 = CtxTransformer(
            d_model=cfg.d_latent,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            n_layers=cfg.n_layers_dec,
            rope_theta=cfg.rope_theta,
            use_sdpa=cfg.use_sdpa,
            gradient_checkpointing=cfg.gradient_checkpointing
        )
        self.lm_head = nn.Linear(cfg.d_latent, cfg.vocab_size, bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.dec_embed.weight
        
        # Loss modules
        self.latent_loss_fn = GaussianLatentLoss()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def encode(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bottom-up encode: tokens -> L1 latents -> L2 latents
        
        Args:
            input_ids: [B, T] token ids (T divisible by C1*C2)
        
        Returns:
            x1: [B, T/C1, d_latent] level-1 latents
            x2: [B, T/(C1*C2), d_latent] level-2 latents
        """
        x = self.enc_embed(input_ids)  # [B, T, d_embed_enc]
        
        # Level 1
        x1 = self.enc_chunk1(x)        # [B, T/C1, d_latent]
        x1 = self.enc_ctx1(x1, is_causal=True)
        
        # Level 2
        x2 = self.enc_chunk2(x1)       # [B, T/(C1*C2), d_latent]
        x2 = self.enc_ctx2(x2, is_causal=True)
        
        return x1, x2
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                return_latents: bool = False) -> Dict[str, torch.Tensor]:
        """
        Training forward pass with teacher forcing.
        
        Args:
            input_ids: [B, T] token ids
            labels: [B, T] target token ids (usually same as input_ids for LM)
            return_latents: If True, include latent tensors in output
        
        Returns:
            Dict with loss, loss_latent, loss_lm, logits, and optionally latents
        """
        B, T = input_ids.shape
        cfg = self.cfg
        
        # Pad to block size if needed
        block = cfg.C1 * cfg.C2
        if T % block != 0:
            pad = block - (T % block)
            input_ids = F.pad(input_ids, (0, pad), value=cfg.pad_token_id or 0)
            if labels is not None:
                labels = F.pad(labels, (0, pad), value=-100)
            T = input_ids.size(1)
        
        # Encode to get target latents
        x1, x2 = self.encode(input_ids)
        M2 = x2.size(1)  # Number of level-2 units
        M1 = x1.size(1)  # Number of level-1 units
        
        # =====================================================================
        # (A) Level-2 -> Level-1 latent reconstruction loss
        # =====================================================================
        
        # Target: x1 grouped into chunks of C2
        # IMPORTANT: Detach targets to prevent encoder from learning to produce
        # "easily predictable" latents. This forces the encoder to learn good
        # representations rather than colluding with the decoder.
        x1_chunks = x1.detach().view(B, M2, cfg.C2, cfg.d_latent)  # [B, M2, C2, D]
        
        # Previous level-2 latent (start token for g=0)
        # DETACH x2 conditioning to prevent encoder-decoder collusion
        prev_l2 = torch.cat([
            self.start_latent_l2.view(1, 1, -1).expand(B, 1, -1),
            x2[:, :-1, :].detach(),
        ], dim=1)  # [B, M2, D]
        
        # Conditioning prefix from converter
        cond2 = self.dec_conv2(prev_l2)  # [B, M2, R2, D]
        
        # Slot tokens for positions to predict (match dtype of encoded latents)
        slots2 = torch.zeros(B, M2, cfg.C2, cfg.d_latent, device=input_ids.device, dtype=x1.dtype)
        
        # Decoder input: [conditioning ; slots]
        dec_in2 = torch.cat([cond2, slots2], dim=2)  # [B, M2, R2+C2, D]
        dec_in2 = dec_in2.view(B * M2, cfg.R2 + cfg.C2, cfg.d_latent)
        
        # Decode
        dec_out2 = self.dec_ctx2(dec_in2, is_causal=True)  # [B*M2, R2+C2, D]
        pred_h = dec_out2[:, cfg.R2:, :]  # [B*M2, C2, D]
        
        # Gaussian prediction
        pred_mean = self.latent_mean_head(pred_h)      # [B*M2, C2, D]
        pred_logvar = self.latent_logvar_head(pred_h)  # [B*M2, C2, D]
        
        pred_mean = pred_mean.view(B, M2, cfg.C2, cfg.d_latent)
        pred_logvar = pred_logvar.view(B, M2, cfg.C2, cfg.d_latent)
        
        # Latent loss
        if cfg.latent_loss_type == "gaussian":
            loss_latent = self.latent_loss_fn(pred_mean, pred_logvar, x1_chunks)
        else:
            loss_latent = F.mse_loss(pred_mean, x1_chunks)
        
        # =====================================================================
        # (B) Level-2 AR loss (train the latent AR head)
        # =====================================================================
        
        # Train AR head to predict x2[g] from x2[<g]
        if M2 > 1:
            ar_mean, ar_logvar = self.latent_ar_head(x2)
            # Predict x2[1:] from x2[:-1]
            ar_mean_shifted = ar_mean[:, :-1, :]
            ar_logvar_shifted = ar_logvar[:, :-1, :]
            ar_target = x2[:, 1:, :].detach()  # Detach target
            loss_latent_ar = self.latent_loss_fn(ar_mean_shifted, ar_logvar_shifted, ar_target)
            loss_latent = loss_latent + loss_latent_ar
        
        # =====================================================================
        # (C) Token LM loss within chunks
        # =====================================================================
        
        # Group tokens into chunks
        tokens = input_ids.view(B, M1, cfg.C1)  # [B, M1, C1]
        tok_emb = self.dec_embed(tokens)         # [B, M1, C1, D]
        
        # Previous level-1 latent
        # DETACH x1 conditioning to prevent encoder-decoder collusion
        prev_l1 = torch.cat([
            self.start_latent_l1.view(1, 1, -1).expand(B, 1, -1),
            x1[:, :-1, :].detach(),
        ], dim=1)  # [B, M1, D]
        
        # Conditioning prefix
        cond1 = self.dec_conv1(prev_l1)  # [B, M1, R1, D]
        
        # Decoder input: [conditioning ; token_embeddings]
        dec_in1 = torch.cat([cond1, tok_emb], dim=2)  # [B, M1, R1+C1, D]
        dec_in1 = dec_in1.view(B * M1, cfg.R1 + cfg.C1, cfg.d_latent)
        
        # Decode
        dec_out1 = self.dec_ctx1(dec_in1, is_causal=True)  # [B*M1, R1+C1, D]
        token_h = dec_out1[:, cfg.R1:, :]  # [B*M1, C1, D]
        
        # LM head
        logits = self.lm_head(token_h)  # [B*M1, C1, vocab]
        logits = logits.view(B, M1 * cfg.C1, cfg.vocab_size)
        
        # Cross-entropy loss
        loss_lm = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Handle EOS masking
            if cfg.eos_token_id is not None:
                # Create mask that zeros out loss after EOS
                eos_mask = self._create_eos_mask(shift_labels, cfg.eos_token_id)
                shift_labels = shift_labels.clone()
                shift_labels[~eos_mask] = -100
            
            loss_lm = F.cross_entropy(
                shift_logits.view(-1, cfg.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        # Combined loss
        loss = cfg.lambda_latent * loss_latent
        if loss_lm is not None:
            loss = loss + cfg.lambda_lm * loss_lm
        
        out = {
            "loss": loss,
            "loss_latent": loss_latent,
            "logits": logits,
        }
        if loss_lm is not None:
            out["loss_lm"] = loss_lm
        
        if return_latents:
            out["x1"] = x1
            out["x2"] = x2
        
        return out
    
    def _create_eos_mask(self, labels: torch.Tensor, eos_id: int) -> torch.Tensor:
        """Create mask that is True before (and including) first EOS."""
        B, T = labels.shape
        is_eos = labels == eos_id
        
        # Cumsum to find positions after first EOS
        eos_cumsum = is_eos.cumsum(dim=1)
        
        # Mask is True where cumsum <= 1 (before or at first EOS)
        # But we want to include the EOS itself in training
        mask = eos_cumsum <= 1
        
        return mask
    
    def get_last_latents(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode and return the last latent at each level.
        Useful for starting generation.
        
        Returns:
            last_l1: [B, d_latent]
            last_l2: [B, d_latent]
        """
        x1, x2 = self.encode(input_ids)
        return x1[:, -1, :], x2[:, -1, :]