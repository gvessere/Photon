"""
PHOTON Inference Module

Recursive generation without bottom-up re-encoding:
1. Use decoder-side reconstructions to update the coarse stream
2. Cascade down through converters
3. Generate tokens chunk by chunk
"""

from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F

from .model import PhotonLM
from .config import PhotonConfig


@torch.no_grad()
def generate_photon(
    model: PhotonLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate tokens using PHOTON's hierarchical latent structure.
    
    Recursive generation:
    1. Encode prompt to get initial latents
    2. Decode top-down to generate tokens and decoder-side L1 reconstructions
    3. Update coarse stream from decoder-side reconstructions (no bottom-up re-encoding)
    
    Args:
        model: PhotonLM model
        input_ids: [B, T] prompt token ids
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature for tokens
        top_k: Top-k sampling (0 to disable)
        top_p: Top-p (nucleus) sampling (1.0 to disable)
        eos_token_id: Stop generation on this token
    
    Returns:
        [B, T + new_tokens] generated sequence
    """
    model.eval()
    cfg = model.cfg
    device = input_ids.device
    B = input_ids.size(0)
    
    # Pad prompt to block size
    T = input_ids.size(1)
    block = cfg.C1 * cfg.C2
    if T % block != 0:
        pad = block - (T % block)
        input_ids = F.pad(input_ids, (0, pad), value=cfg.pad_token_id or 0)
    
    # Initial encoding
    x1, x2 = model.encode(input_ids)
    
    # Current state
    cur_tokens = input_ids.clone()
    
    # Track coarse stream inputs (summaries) for RecGen updates
    a2_history = model.enc_chunk2(x1)  # [B, M2, D]
    prev_l1 = x1[:, -1, :]  # [B, D] - last L1 latent

    # Initialize streaming cache for top-level context encoder
    ctx2_cache = model.enc_ctx2.init_kv_cache()
    last_x2 = None
    for i in range(a2_history.size(1)):
        last_x2, ctx2_cache = model.enc_ctx2.forward_step(a2_history[:, i:i+1, :], ctx2_cache)
    prev_l2 = last_x2[:, -1, :]
    
    new_tokens = []
    tokens_generated = 0
    
    while tokens_generated < max_new_tokens:
        # Generate C1*C2 tokens (one full block) at a time
        block_tokens, l1_latents = generate_one_block(
            model=model,
            prev_l1=prev_l1,
            prev_l2=prev_l2,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )  # [B, C1*C2], [B, C2, D]
        
        new_tokens.append(block_tokens)
        cur_tokens = torch.cat([cur_tokens, block_tokens], dim=1)
        tokens_generated += block_tokens.size(1)
        
        # Check for EOS
        if eos_token_id is not None:
            if (block_tokens == eos_token_id).any():
                break
        
        # Recursive update: use decoder-side L1 reconstructions to update coarse stream
        next_a2 = model.enc_chunk2(l1_latents)  # [B, 1, D]
        a2_history = torch.cat([a2_history, next_a2], dim=1)
        last_x2, ctx2_cache = model.enc_ctx2.forward_step(next_a2, ctx2_cache)
        prev_l2 = last_x2[:, -1, :]
        prev_l1 = l1_latents[:, -1, :]
    
    # Concatenate and truncate
    generated = cur_tokens[:, :input_ids.size(1) + max_new_tokens]
    
    return generated


def generate_one_block(
    model: PhotonLM,
    prev_l1: torch.Tensor,
    prev_l2: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate one full block of C1*C2 tokens.
    
    Uses the hierarchical structure:
    1. Decode L2 -> C2 L1 latents (deterministic)
    2. Decode each L1 -> C1 tokens
    
    Args:
        model: PhotonLM model
        prev_l1: [B, D] previous L1 latent (for first chunk)
        prev_l2: [B, D] previous L2 latent
        temperature: Token sampling temperature
        top_k: Top-k sampling
        top_p: Top-p sampling
    
    Returns:
        ([B, C1*C2] generated tokens, [B, C2, D] L1 latents)
    """
    cfg = model.cfg
    B = prev_l1.size(0)
    device = prev_l1.device
    
    # Step 1: Decode L2 -> C2 L1 latents
    # Condition on prev_l2 using input converter
    cond2 = model.dec_conv2_in(prev_l2)  # [B, R2, D]
    
    # Generate C2 L1 latents autoregressively
    l1_latents = []
    current_l1 = prev_l1
    
    for i in range(cfg.C2):
        # Decode to get next L1 latent
        # Use slots for prediction positions
        slots = torch.zeros(B, 1, cfg.d_latent, device=device)
        
        if i == 0:
            dec_in = torch.cat([cond2, slots], dim=1)  # [B, R2+1, D]
        else:
            # Include previously generated L1 latents
            prev_l1s = torch.stack(l1_latents, dim=1)  # [B, i, D]
            dec_in = torch.cat([cond2, prev_l1s, slots], dim=1)  # [B, R2+i+1, D]
        
        dec_out = model.dec_ctx2(dec_in, is_causal=True)
        
        # Output projection to L1 latent
        h = dec_out[:, -1, :]  # [B, D]
        next_l1 = model.dec_proj2_out(h)
        
        l1_latents.append(next_l1)
        current_l1 = next_l1
    
    # l1_latents: list of [B, D], length C2
    l1_latents = torch.stack(l1_latents, dim=1)  # [B, C2, D]
    
    # Step 2: Decode each L1 latent -> C1 tokens
    all_tokens = []
    prev_l1_for_chunk = prev_l1  # Start with the passed-in prev_l1
    
    for j in range(cfg.C2):
        # Generate C1 tokens from L1 latent j
        chunk_tokens = generate_token_chunk(
            model=model,
            prev_l1=prev_l1_for_chunk,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )  # [B, C1]
        all_tokens.append(chunk_tokens)
        
        # Update prev_l1 for next chunk
        prev_l1_for_chunk = l1_latents[:, j, :]
    
    # Concatenate all chunks
    block_tokens = torch.cat(all_tokens, dim=1)  # [B, C1*C2]
    
    return block_tokens, l1_latents


def generate_token_chunk(
    model: PhotonLM,
    prev_l1: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> torch.Tensor:
    """
    Generate C1 tokens conditioned on previous L1 latent.
    
    Autoregressive within the chunk.
    
    Args:
        model: PhotonLM model
        prev_l1: [B, D] conditioning latent
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Top-p sampling
    
    Returns:
        [B, C1] generated tokens
    """
    cfg = model.cfg
    B = prev_l1.size(0)
    device = prev_l1.device
    
    # Get conditioning prefix
    cond1 = model.dec_conv1(prev_l1)  # [B, R1, D]
    
    # Generate tokens autoregressively
    chunk_tokens = []
    
    for i in range(cfg.C1):
        # Build decoder input
        if len(chunk_tokens) == 0:
            dec_in = cond1  # [B, R1, D]
        else:
            # Embed previous tokens
            prev_tok = torch.stack(chunk_tokens, dim=1)  # [B, i]
            tok_emb = model.dec_embed(prev_tok)  # [B, i, D]
            dec_in = torch.cat([cond1, tok_emb], dim=1)  # [B, R1+i, D]
        
        # Decode
        dec_out = model.dec_ctx1(dec_in, is_causal=True)
        h_last = dec_out[:, -1, :]  # [B, D]
        
        # LM head
        logits = model.lm_head(h_last)  # [B, vocab]
        
        # Sample next token
        next_token = sample_token(logits, temperature, top_k, top_p)
        chunk_tokens.append(next_token)
    
    return torch.stack(chunk_tokens, dim=1)  # [B, C1]


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> torch.Tensor:
    """
    Sample a token from logits with temperature, top-k, and top-p.
    
    Args:
        logits: [B, vocab] unnormalized logits
        temperature: Sampling temperature (0 = greedy)
        top_k: Keep only top-k tokens (0 = disabled)
        top_p: Keep tokens with cumulative prob < top_p (1.0 = disabled)
    
    Returns:
        [B] sampled token ids
    """
    if temperature == 0:
        return logits.argmax(dim=-1)
    
    logits = logits / temperature
    
    # Top-k filtering
    if top_k > 0 and top_k < logits.size(-1):
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float('-inf')
    
    # Top-p filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        
        # Remove tokens with cumulative prob above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift to keep first token above threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        
        # Scatter back
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
    
    # Sample
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def generate_with_kv_cache(
    model: PhotonLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 50,
) -> torch.Tensor:
    """
    Generate with KV cache optimization (placeholder for future implementation).
    
    PHOTON's design reduces KV cache needs by chunking, but within-chunk
    decoding can still benefit from caching.
    
    For now, this is a simple wrapper around generate_photon.
    """
    # TODO: Implement proper KV caching for chunk-local decoders
    return generate_photon(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
