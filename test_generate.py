#!/usr/bin/env python
"""
Generation for PHOTON and Baseline models (local or W&B checkpoint).

Usage:
    # Local PHOTON
    python test_generate.py --checkpoint checkpoints_photon/photon_1000.pt

    # W&B by run id (artifact name defaults to photon-<run_id>:latest)
    python test_generate.py --wandb-run-id <id> --wandb-project photon

    # W&B full artifact reference
    python test_generate.py --wandb-artifact entity/project/photon-abc123:latest

    # PHOTON: full L2 encode on prompt then block decode (default), or re-encode path
    python test_generate.py --checkpoint ckpt.pt --inference-mode prefill
    python test_generate.py --checkpoint ckpt.pt --inference-mode reencode_blocks
    python test_generate.py --checkpoint ckpt.pt --inference-mode reencode_tokens

    # Low memory
    python test_generate.py --checkpoint ckpt.pt --cpu
"""

import argparse
import sys
from typing import Optional

import torch
from transformers import AutoTokenizer

sys.path.insert(0, ".")

from photon import PhotonConfig, PhotonLM
from photon.inference import generate_photon
from baseline import BaselineConfig, BaselineLM
from train_utils import download_wandb_checkpoint_for_inference


def _normalize_inference_mode(mode: str) -> str:
    """
    Canonical PHOTON modes:
      prefill — full encoder on prompt, then block decode + RecGen (default)
      reencode_blocks — same decoder, but full bottom-up encode after each block
      reencode_tokens — one token at a time, full model() forward each step
    """
    m = mode.strip().lower()
    aliases = {
        "recgen": "prefill",
        "lightning": "prefill",
        "encode_then_generate": "prefill",
        "block_reencode": "reencode_blocks",
        "ar_reencode": "reencode_tokens",
    }
    return aliases.get(m, m)


def _resolve_checkpoint_path(args: argparse.Namespace) -> str:
    sources = [args.checkpoint, args.wandb_artifact, args.wandb_run_id]
    n = sum(1 for s in sources if s)
    if n != 1:
        raise SystemExit(
            "Specify exactly one of: --checkpoint, --wandb-artifact, or --wandb-run-id"
        )
    if args.checkpoint:
        return args.checkpoint
    return download_wandb_checkpoint_for_inference(
        artifact_ref=args.wandb_artifact,
        run_id=args.wandb_run_id,
        artifact_name=args.wandb_artifact_name,
        artifact_alias=args.wandb_artifact_alias,
        artifact_file=args.wandb_artifact_file,
        entity=args.wandb_entity,
        project=args.wandb_project,
        wandb_project=args.wandb_project or "photon",
        resume_prefix=args.wandb_resume_prefix,
        cache_dir=args.wandb_cache_dir,
    )


def _generate_ar_reencode(
    model,
    model_type: str,
    cfg,
    block_size: Optional[int],
    input_ids: torch.Tensor,
    max_tokens: int,
    temperature: float,
    device: str,
    dtype: torch.dtype,
    tokenizer,
):
    """Token autoregressive decode with a full forward pass each new token."""
    generated = input_ids.clone()
    generated_tokens: list[int] = []

    with torch.inference_mode():
        for _ in range(max_tokens):
            if model_type == "photon" and block_size:
                curr_len = generated.size(1)
                if curr_len % block_size != 0:
                    pad = block_size - (curr_len % block_size)
                    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                    generated = torch.cat(
                        [
                            torch.full((1, pad), pad_id, device=device, dtype=torch.long),
                            generated,
                        ],
                        dim=1,
                    )

            with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
                out = model(generated)

            logits = out["logits"]
            next_logits = logits[0, -1, :].float() / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated_tokens.append(next_token.item())
            token_str = tokenizer.decode([next_token.item()])
            print(token_str, end="", flush=True)

            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

            if model_type == "baseline" and hasattr(cfg, "max_seq_len"):
                if generated.size(1) > cfg.max_seq_len:
                    generated = generated[:, -cfg.max_seq_len :]

    return generated, generated_tokens


def _generate_photon_mode(
    model: PhotonLM,
    input_ids: torch.Tensor,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    tokenizer,
    mode: str,
    device: str,
    dtype: torch.dtype,
):
    """Block-wise PHOTON: prefill + RecGen, or full re-encode after each block."""
    reencode_after_each_block = mode == "reencode_blocks"
    with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
        out_ids = generate_photon(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            reencode_after_each_block=reencode_after_each_block,
        )

    # Decode only newly generated tail for streaming-style print
    prompt_len = input_ids.size(1)
    new_part = out_ids[0, prompt_len:]
    for tid in new_part.tolist():
        print(tokenizer.decode([tid]), end="", flush=True)

    new_tokens = new_part.numel()
    return out_ids, list(new_part.tolist())


def main():
    parser = argparse.ArgumentParser(
        description="Generate text from a PHOTON or Baseline checkpoint (local or W&B)."
    )
    src = parser.add_argument_group("checkpoint source (pick one)")
    src.add_argument("--checkpoint", type=str, default=None, help="Path to a local .pt file")
    src.add_argument(
        "--wandb-artifact",
        type=str,
        default=None,
        help=(
            "Artifact ref: entity/project/name:alias_or_version. "
            "Pasted artifact:///entity/project/name:vN URIs from the UI are also accepted."
        ),
    )
    src.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="W&B run id; downloads artifact named {wandb-resume-prefix}-{run_id}",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project (for resolving run id / entity default)",
    )
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity")
    parser.add_argument(
        "--wandb-artifact-name",
        type=str,
        default=None,
        help="Override artifact collection name (default: {wandb-resume-prefix}-{run_id})",
    )
    parser.add_argument(
        "--wandb-artifact-alias",
        type=str,
        default="latest",
        help="Artifact alias or version (default: latest)",
    )
    parser.add_argument(
        "--wandb-artifact-file",
        type=str,
        default=None,
        help="Specific .pt filename inside the artifact (default: highest step)",
    )
    parser.add_argument(
        "--wandb-resume-prefix",
        type=str,
        default="photon",
        help="Default artifact name prefix when using --wandb-run-id (photon or baseline)",
    )
    parser.add_argument(
        "--wandb-cache-dir",
        type=str,
        default=".wandb_inference_artifacts",
        help="Directory for downloaded W&B artifacts",
    )

    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="PHOTON prefill / reencode_blocks only",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="PHOTON prefill / reencode_blocks only",
    )
    parser.add_argument(
        "--inference-mode",
        type=str,
        default="prefill",
        help=(
            "PHOTON: prefill (full L2 encode on prompt, then block decode + RecGen; "
            "aliases: recgen, lightning); "
            "reencode_blocks (re-run full encoder on all tokens after each block; "
            "alias: block_reencode); "
            "reencode_tokens (token AR with full model() each step; alias: ar_reencode). "
            "Baseline: reencode_tokens only."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="HF tokenizer id (must match training)",
    )
    parser.add_argument("--cpu", action="store_true", help="Run on CPU")
    parser.add_argument("--fp32", action="store_true", help="Use fp32 instead of fp16")

    args = parser.parse_args()
    mode = _normalize_inference_mode(args.inference_mode)

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.fp32 or args.cpu else torch.float16

    ckpt_path = _resolve_checkpoint_path(args)

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Loading checkpoint: {ckpt_path}")

    torch.serialization.add_safe_globals([PhotonConfig, BaselineConfig])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    cfg = ckpt.get("config")
    if cfg is None:
        if "baseline" in ckpt_path.lower():
            print("No config found, inferring Baseline from path")
            cfg = BaselineConfig()
            model_type = "baseline"
        else:
            print("No config found, inferring PHOTON from path")
            cfg = PhotonConfig()
            model_type = "photon"
    elif isinstance(cfg, BaselineConfig):
        model_type = "baseline"
        print("Detected: Baseline Transformer")
    elif isinstance(cfg, PhotonConfig):
        model_type = "photon"
        print("Detected: PHOTON")
    else:
        if hasattr(cfg, "C1") and hasattr(cfg, "C2"):
            model_type = "photon"
            print("Detected: PHOTON (from attributes)")
        else:
            model_type = "baseline"
            print("Detected: Baseline (from attributes)")

    if model_type == "baseline" and mode not in ("reencode_tokens",):
        print(
            f"Note: inference-mode '{args.inference_mode}' is only defined for PHOTON; "
            "using reencode_tokens for baseline."
        )
        mode = "reencode_tokens"

    if model_type == "photon" and mode not in (
        "prefill",
        "reencode_blocks",
        "reencode_tokens",
    ):
        raise SystemExit(
            f"Unknown inference-mode for PHOTON: {args.inference_mode}. "
            "Use prefill, reencode_blocks, or reencode_tokens "
            "(aliases: recgen/lightning, block_reencode, ar_reencode)."
        )

    print(f"Inference mode: {mode}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    cfg.vocab_size = len(tokenizer)
    cfg.eos_token_id = tokenizer.eos_token_id
    cfg.pad_token_id = tokenizer.pad_token_id

    print(f"Creating {model_type} model...")
    if model_type == "photon":
        model = PhotonLM(cfg)
        block_size = cfg.C1 * cfg.C2
    else:
        model = BaselineLM(cfg)
        block_size = None

    state_dict = ckpt.get("model", ckpt.get("model_state_dict", {}))
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    empty_count = sum(1 for v in state_dict.values() if v.numel() == 0)
    if empty_count > 10:
        print(f"ERROR: Checkpoint has {empty_count} empty tensors (bad ZeRO-3 save)")
        return

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    del ckpt, state_dict
    if device == "cuda":
        torch.cuda.empty_cache()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.1f}M params")

    if device == "cuda":
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory: {mem:.2f} GB")

    print(f"\n{'=' * 60}")
    print(f"Prompt: {args.prompt}")
    print(f"{'=' * 60}\n")

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)

    if model_type == "photon" and block_size:
        if input_ids.size(1) < block_size:
            pad_len = block_size - input_ids.size(1)
            pad_id = (
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            )
            input_ids = torch.cat(
                [
                    torch.full((1, pad_len), pad_id, device=device, dtype=torch.long),
                    input_ids,
                ],
                dim=1,
            )

    print("Generating...\n")

    if model_type == "photon" and mode in ("prefill", "reencode_blocks"):
        generated, generated_tokens = _generate_photon_mode(
            model=model,
            input_ids=input_ids,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            tokenizer=tokenizer,
            mode=mode,
            device=device,
            dtype=dtype,
        )
    else:
        generated, generated_tokens = _generate_ar_reencode(
            model=model,
            model_type=model_type,
            cfg=cfg,
            block_size=block_size,
            input_ids=input_ids,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            device=device,
            dtype=dtype,
            tokenizer=tokenizer,
        )

    print(f"\n\n{'=' * 60}")
    print(f"Generated {len(generated_tokens)} tokens")

    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"\nFull text:\n{output_text}")


if __name__ == "__main__":
    main()
