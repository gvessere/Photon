#!/usr/bin/env python
"""
Generation test for PHOTON and Baseline models

Usage:
    # PHOTON
    python test_generate.py --checkpoint checkpoints_photon/photon_1000.pt
    
    # Baseline
    python test_generate.py --checkpoint checkpoints_baseline/baseline_1000.pt
    
    # Low memory mode
    python test_generate.py --checkpoint checkpoints_photon/photon_1000.pt --cpu
"""

import argparse
import torch
from transformers import AutoTokenizer

import sys
sys.path.insert(0, '.')

from photon import PhotonConfig, PhotonLM
from baseline import BaselineConfig, BaselineLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint (auto-detects model type)")
    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--cpu", action="store_true", help="Run on CPU (slower but less memory)")
    parser.add_argument("--fp32", action="store_true", help="Use fp32 instead of fp16")
    args = parser.parse_args()
    
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.fp32 or args.cpu else torch.float16
    
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Loading checkpoint: {args.checkpoint}")
    
    # Add both config types to safe globals
    torch.serialization.add_safe_globals([PhotonConfig, BaselineConfig])
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    
    # Detect model type from config
    cfg = ckpt.get("config")
    if cfg is None:
        # Try to infer from checkpoint path
        if "baseline" in args.checkpoint.lower():
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
        # Check config attributes to determine type
        if hasattr(cfg, 'C1') and hasattr(cfg, 'C2'):
            model_type = "photon"
            print("Detected: PHOTON (from attributes)")
        else:
            model_type = "baseline"
            print("Detected: Baseline (from attributes)")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    cfg.vocab_size = len(tokenizer)
    cfg.eos_token_id = tokenizer.eos_token_id
    cfg.pad_token_id = tokenizer.pad_token_id
    
    # Create appropriate model
    print(f"Creating {model_type} model...")
    if model_type == "photon":
        model = PhotonLM(cfg)
        block_size = cfg.C1 * cfg.C2
    else:
        model = BaselineLM(cfg)
        block_size = None  # Baseline doesn't need block alignment
    
    # Load weights
    state_dict = ckpt.get("model", ckpt.get("model_state_dict", {}))
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # Check for bad checkpoint
    empty_count = sum(1 for v in state_dict.values() if v.numel() == 0)
    if empty_count > 10:
        print(f"ERROR: Checkpoint has {empty_count} empty tensors (bad ZeRO-3 save)")
        return
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    # Free checkpoint memory
    del ckpt, state_dict
    if device == "cuda":
        torch.cuda.empty_cache()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.1f}M params")
    
    if device == "cuda":
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory: {mem:.2f} GB")
    
    print(f"\n{'='*60}")
    print(f"Prompt: {args.prompt}")
    print(f"{'='*60}\n")
    
    # Tokenize prompt
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    
    # For PHOTON, pad to block size
    if model_type == "photon" and block_size:
        if input_ids.size(1) < block_size:
            pad_len = block_size - input_ids.size(1)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            input_ids = torch.cat([
                torch.full((1, pad_len), pad_id, device=device, dtype=torch.long),
                input_ids
            ], dim=1)
    
    # Generate
    print("Generating...\n")
    generated = input_ids.clone()
    generated_tokens = []
    
    with torch.inference_mode():
        for i in range(args.max_tokens):
            # For PHOTON, ensure length is multiple of block_size
            if model_type == "photon" and block_size:
                curr_len = generated.size(1)
                if curr_len % block_size != 0:
                    pad = block_size - (curr_len % block_size)
                    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                    generated = torch.cat([
                        torch.full((1, pad), pad_id, device=device, dtype=torch.long),
                        generated
                    ], dim=1)
            
            # Forward pass
            with torch.autocast(device_type=device, dtype=dtype, enabled=(device=="cuda")):
                out = model(generated)
            
            logits = out["logits"]
            
            # Get next token prediction
            next_logits = logits[0, -1, :].float() / args.temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_tokens.append(next_token.item())
            
            # Print token as we go
            token_str = tokenizer.decode([next_token.item()])
            print(token_str, end="", flush=True)
            
            # Append
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Stop on EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # For baseline, limit context window
            if model_type == "baseline" and hasattr(cfg, 'max_seq_len'):
                if generated.size(1) > cfg.max_seq_len:
                    generated = generated[:, -cfg.max_seq_len:]
    
    print(f"\n\n{'='*60}")
    print(f"Generated {len(generated_tokens)} tokens")
    
    # Show full output
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"\nFull text:\n{output_text}")


if __name__ == "__main__":
    main()
