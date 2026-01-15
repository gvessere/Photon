#!/usr/bin/env python
"""
Quick generation test for PHOTON

Usage:
    python test_generate.py --checkpoint checkpoints_photon/photon_1000.pt
    python test_generate.py --checkpoint checkpoints_photon/photon_1000.pt --cpu  # Low memory
"""

import argparse
import torch
from transformers import AutoTokenizer

import sys
sys.path.insert(0, '.')

from photon import PhotonConfig, PhotonLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints_photon/photon_1000.pt")
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
    
    # Load checkpoint to CPU first to save GPU memory
    torch.serialization.add_safe_globals([PhotonConfig])
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    
    # Get config
    cfg = ckpt.get("config", PhotonConfig())
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    cfg.vocab_size = len(tokenizer)
    cfg.eos_token_id = tokenizer.eos_token_id
    cfg.pad_token_id = tokenizer.pad_token_id
    
    # Create model directly on device with correct dtype
    print("Creating model...")
    model = PhotonLM(cfg)
    
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
    
    # Move to device and convert dtype
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
    
    # Pad to block size
    block_size = cfg.C1 * cfg.C2
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
            # Ensure length is multiple of block_size
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
    
    print(f"\n\n{'='*60}")
    print(f"Generated {len(generated_tokens)} tokens")
    
    # Show full output
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"\nFull text:\n{output_text}")


if __name__ == "__main__":
    main()
