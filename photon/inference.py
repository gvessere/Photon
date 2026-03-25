"""
PHOTON Inference Module — KV-cached generation

Performance-critical paths use ``forward_prefill`` + ``forward_step`` so that:

* The L1 encoder (``enc_ctx1``) is prefilled once on the prompt and extended by
  a single position per C1-token chunk — across the entire generation, not just
  within a single block.
* The token decoder (``dec_ctx1``) prefills the R1 conditioning vectors once per
  chunk, then single-steps for each of the C1 autoregressive tokens.

Correctness is identical to the previous "re-encode after each chunk" approach
because ``enc_ctx1`` is causal — appending a new position never changes the
representations at earlier positions.
"""

from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .model import PhotonLM
from .config import PhotonConfig


# ---------------------------------------------------------------------------
# Initialisation: full encode that also builds the enc_ctx1 KV cache
# ---------------------------------------------------------------------------

def _init_generation_state(
    model: PhotonLM,
    token_ids: torch.Tensor,
) -> Tuple[torch.Tensor, list, torch.Tensor, torch.Tensor, list]:
    """
    Full bottom-up encode of ``token_ids`` returning everything needed to start
    block-wise generation:

    * ``enc1_kv`` — KV cache for ``enc_ctx1`` (incremental L1 updates)
    * ``prev_l1_vec`` — L1 vector of the last C1 window (conditioner for next chunk)
    * ``prev_l2`` — L2 vector for the first block's L2 decoder
    * ``ctx2_cache`` — KV cache for the L2 context encoder stream
    * ``a2_history`` — accumulated L2 chunk summaries

    This replaces the old ``_streaming_l2_state_from_full_encode``.
    """
    emb = model.enc_embed(token_ids)           # [B, T, d_embed]
    x1_raw = model.enc_chunk1(emb)             # [B, T/C1, d_latent]
    x1, enc1_kv = model.enc_ctx1.forward_prefill(x1_raw, is_causal=True)

    prev_l1_vec = x1[:, -1, :]                # [B, D]

    a2_history = model.enc_chunk2(x1)          # [B, T/(C1*C2), D]
    ctx2_cache = model.enc_ctx2.init_kv_cache()
    last_x2 = None
    for i in range(a2_history.size(1)):
        last_x2, ctx2_cache = model.enc_ctx2.forward_step(
            a2_history[:, i : i + 1, :], ctx2_cache,
        )
    prev_l2 = last_x2[:, -1, :]               # [B, D]

    return a2_history, enc1_kv, prev_l1_vec, prev_l2, ctx2_cache


# ---------------------------------------------------------------------------
# Token-chunk generation (KV-cached dec_ctx1)
# ---------------------------------------------------------------------------

def generate_token_chunk(
    model: PhotonLM,
    prev_l1: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> torch.Tensor:
    """
    Generate C1 tokens conditioned on the previous L1 latent.

    Uses ``forward_prefill`` on the R1 conditioning vectors, then
    ``forward_step`` for each autoregressive token — O(R1 + C1) work instead
    of O(R1 * C1 + C1^2 / 2).
    """
    cfg = model.cfg

    cond1 = model.dec_conv1(prev_l1)  # [B, R1, D]

    dec_out, dec1_kv = model.dec_ctx1.forward_prefill(cond1, is_causal=True)

    chunk_tokens: List[torch.Tensor] = []
    for i in range(cfg.C1):
        if i == 0:
            h = dec_out[:, -1, :]
        else:
            tok_emb = model.dec_embed(chunk_tokens[-1]).unsqueeze(1)  # [B, 1, D]
            step_out, dec1_kv = model.dec_ctx1.forward_step(tok_emb, dec1_kv)
            h = step_out[:, 0, :]

        logits = model.lm_head(h)
        next_token = sample_token(logits, temperature, top_k, top_p)
        chunk_tokens.append(next_token)

    return torch.stack(chunk_tokens, dim=1)  # [B, C1]


# ---------------------------------------------------------------------------
# Block generation (incremental enc_ctx1)
# ---------------------------------------------------------------------------

def generate_one_block(
    model: PhotonLM,
    prev_l2: torch.Tensor,
    prev_l1_vec: torch.Tensor,
    enc1_kv: list,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor, list, torch.Tensor]:
    """
    Generate one full block of ``C1 * C2`` tokens.

    1. L2 decoder produces C2 predicted-L1 latents (used only for RecGen).
    2. For each of C2 chunks:
       a. Decode C1 tokens conditioned on ``prev_l1_vec``.
       b. Embed + chunk the new tokens → 1 new L1 position →
          ``enc_ctx1.forward_step`` (O(1) new compute, attending to the
          full cached prefix).  Update ``prev_l1_vec`` for the next chunk.

    Returns ``(block_tokens, l1_latents, enc1_kv, prev_l1_vec)``
    """
    cfg = model.cfg
    B = prev_l2.size(0)
    device = prev_l2.device

    # --- L2 decoder (unchanged) ---
    cond2 = model.dec_conv2_in(prev_l2)
    slots2 = torch.zeros(B, cfg.C2, cfg.d_latent, device=device, dtype=cond2.dtype)
    dec_in2 = torch.cat([cond2, slots2], dim=1)
    dec_out2 = model.dec_ctx2(dec_in2, is_causal=True)
    pred_h = dec_out2[:, cfg.R2:, :]
    l1_latents = model.dec_proj2_out(pred_h)   # [B, C2, D]

    # --- C2 token chunks with incremental L1 conditioning ---
    all_chunks: List[torch.Tensor] = []

    for _j in range(cfg.C2):
        chunk_tokens = generate_token_chunk(
            model, prev_l1_vec,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )  # [B, C1]
        all_chunks.append(chunk_tokens)

        new_emb = model.enc_embed(chunk_tokens)        # [B, C1, d_embed]
        new_l1_raw = model.enc_chunk1(new_emb)         # [B, 1, d_latent]
        new_l1_out, enc1_kv = model.enc_ctx1.forward_step(new_l1_raw, enc1_kv)
        prev_l1_vec = new_l1_out[:, 0, :]

    block_tokens = torch.cat(all_chunks, dim=1)        # [B, C1*C2]
    return block_tokens, l1_latents, enc1_kv, prev_l1_vec


# ---------------------------------------------------------------------------
# Top-level generation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_photon(
    model: PhotonLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    eos_token_id: Optional[int] = None,
    reencode_after_each_block: bool = False,
) -> torch.Tensor:
    """
    Generate tokens using PHOTON's hierarchical decoder.

    The enc_ctx1 KV cache is maintained across the entire generation so the
    prompt encoder is never re-run.  Within each block the cache grows by C2
    positions (one per chunk).

    ``reencode_after_each_block=True`` rebuilds the *L2* state from a fresh
    bottom-up encode after every block (the enc_ctx1 cache is also rebuilt so
    positions stay consistent).
    """
    model.eval()
    cfg = model.cfg
    B = input_ids.size(0)

    # Pad prompt to block boundary
    T = input_ids.size(1)
    block = cfg.C1 * cfg.C2
    if T % block != 0:
        pad = block - (T % block)
        input_ids = F.pad(input_ids, (0, pad), value=cfg.pad_token_id or 0)

    cur_tokens = input_ids.clone()

    a2_history, enc1_kv, prev_l1_vec, prev_l2, ctx2_cache = (
        _init_generation_state(model, cur_tokens)
    )

    tokens_generated = 0

    while tokens_generated < max_new_tokens:
        block_tokens, l1_latents, enc1_kv, prev_l1_vec = generate_one_block(
            model=model,
            prev_l2=prev_l2,
            prev_l1_vec=prev_l1_vec,
            enc1_kv=enc1_kv,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        cur_tokens = torch.cat([cur_tokens, block_tokens], dim=1)
        tokens_generated += block_tokens.size(1)

        if eos_token_id is not None and (block_tokens == eos_token_id).any():
            break

        if reencode_after_each_block:
            a2_history, enc1_kv, prev_l1_vec, prev_l2, ctx2_cache = (
                _init_generation_state(model, cur_tokens)
            )
        else:
            # RecGen: extend L2 stream from decoder-predicted L1 latents
            next_a2 = model.enc_chunk2(l1_latents)  # [B, 1, D]
            a2_history = torch.cat([a2_history, next_a2], dim=1)
            last_x2, ctx2_cache = model.enc_ctx2.forward_step(next_a2, ctx2_cache)
            prev_l2 = last_x2[:, -1, :]

    return cur_tokens[:, : input_ids.size(1) + max_new_tokens]


# ---------------------------------------------------------------------------
# Sampling utility
# ---------------------------------------------------------------------------

def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> torch.Tensor:
    """Sample a single token from logits with temperature / top-k / top-p."""
    if temperature == 0:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    if top_k > 0 and top_k < logits.size(-1):
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove,
        )
        logits[indices_to_remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ---------------------------------------------------------------------------
# Legacy / convenience wrapper
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_kv_cache(
    model: PhotonLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 50,
) -> torch.Tensor:
    """Convenience alias — ``generate_photon`` already uses KV caching."""
    return generate_photon(
        model=model,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        reencode_after_each_block=False,
    )
