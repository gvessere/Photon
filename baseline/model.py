"""
Baseline Transformer (GPT-style decoder-only)

A clean, standard transformer implementation for comparison with PHOTON.
Features:
- RoPE positional encoding
- SwiGLU MLP
- RMSNorm
- SDPA attention (FlashAttention backend)
- Gradient checkpointing support
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class BaselineConfig:
    """Configuration for baseline transformer."""
    vocab_size: int = 32000
    d_model: int = 1024
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 2816  # ~2.75x d_model for SwiGLU
    max_seq_len: int = 2048
    
    # RoPE
    rope_theta: float = 10000.0
    
    # Training
    gradient_checkpointing: bool = True
    use_sdpa: bool = True
    
    # Tokens
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    
    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


# =============================================================================
# RoPE
# =============================================================================

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding."""
    
    def __init__(self, dim: int, theta: float = 10000.0, max_seq_len: int = 8192):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
    
    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if seq_len > self._seq_len_cached or self._cos_cached is None:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]
        self._update_cache(seq_len, q.device, q.dtype)
        
        cos = self._cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self._sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed
    
    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)


# =============================================================================
# Attention
# =============================================================================

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and SDPA."""
    
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.use_sdpa = cfg.use_sdpa
        
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = RotaryEmbedding(cfg.d_head, cfg.rope_theta, cfg.max_seq_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # Apply RoPE
        q, k = self.rope(q, k)
        
        # Attention
        if self.use_sdpa:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        else:
            scale = 1.0 / math.sqrt(self.d_head)
            att = torch.matmul(q, k.transpose(-2, -1)) * scale
            mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
            att = F.softmax(att + mask, dim=-1)
            y = torch.matmul(att, v)
        
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(y)


# =============================================================================
# MLP (SwiGLU)
# =============================================================================

class MLP(nn.Module):
    """SwiGLU MLP."""
    
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# =============================================================================
# Transformer Block
# =============================================================================

class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""
    
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.ln1 = nn.RMSNorm(cfg.d_model)
        self.ln2 = nn.RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# =============================================================================
# Baseline LM
# =============================================================================

class BaselineLM(nn.Module):
    """
    Baseline decoder-only transformer (GPT-style).
    
    A clean baseline for comparison with PHOTON.
    """
    
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg
        
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.embed.weight
        
        self.gradient_checkpointing = cfg.gradient_checkpointing
        
        # Initialize
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [B, T] token ids
            labels: [B, T] target token ids
        
        Returns:
            Dict with loss and logits
        """
        x = self.embed(input_ids)
        
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {"loss": loss, "logits": logits}
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Autoregressive generation."""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx_cond = input_ids if input_ids.size(1) <= self.cfg.max_seq_len else input_ids[:, -self.cfg.max_seq_len:]
            
            # Forward
            logits = self(idx_cond)["logits"][:, -1, :]
            
            # Sample
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                
                # Top-k
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # Top-p
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop on EOS
            if self.cfg.eos_token_id is not None and (next_token == self.cfg.eos_token_id).any():
                break
        
        return input_ids


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
