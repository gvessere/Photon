#!/usr/bin/env python
"""
Quick generation test for PHOTON

Usage:
    python test_generate.py --checkpoint checkpoints/checkpoint_1000.pt
"""

import argparse
import torch
from transformers import AutoTokenizer

import sys
sys.path.insert(0, '.')

from photon import PhotonConfig, PhotonLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_1000.pt")
    parser.add_argument("--prompt", type=str, default="The meaning of life is")
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    
    # Get config
    if "config" in ckpt:
        cfg = ckpt["config"]
    else:
        print("No config in checkpoint, using defaults")
        cfg = PhotonConfig()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    cfg.vocab_size = len(tokenizer)
    cfg.eos_token_id = tokenizer.eos_token_id
    cfg.pad_token_id = tokenizer.pad_token_id
    
    # Create model
    print(f"Creating model ({sum(p.numel() for p in PhotonLM(cfg).parameters()) / 1e6:.1f}M params)...")
    model = PhotonLM(cfg)
    
    # Load weights
    state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    # Handle DeepSpeed wrapped state dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=False)
    model.to(args.device)
    model.eval()
    
    print(f"\n{'='*60}")
    print(f"Prompt: {args.prompt}")
    print(f"{'='*60}\n")
    
    # Tokenize prompt
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(args.device)
    
    # Pad to block size
    block_size = cfg.C1 * cfg.C2
    if input_ids.size(1) < block_size:
        pad_len = block_size - input_ids.size(1)
        input_ids = torch.cat([
            torch.full((1, pad_len), tokenizer.pad_token_id or tokenizer.eos_token_id, device=args.device),
            input_ids
        ], dim=1)
    
    # Generate
    print("Generating...")
    with torch.no_grad():
        # Simple autoregressive generation using the decoder
        generated = input_ids.clone()
        
        for _ in range(args.max_tokens):
            # Ensure length is multiple of block_size
            curr_len = generated.size(1)
            if curr_len % block_size != 0:
                pad = block_size - (curr_len % block_size)
                generated = torch.cat([
                    torch.full((1, pad), tokenizer.pad_token_id or 0, device=args.device),
                    generated
                ], dim=1)
            
            # Forward pass
            out = model(generated)
            logits = out["logits"]  # [B, T, vocab]
            
            # Get next token prediction
            next_logits = logits[0, -1, :] / args.temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Stop on EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    
    print(f"Generated text:\n{'-'*60}")
    print(output_text)
    print(f"{'-'*60}\n")
    
    # Also show raw tokens to check for repetition/garbage
    print("Raw token IDs (last 50):", generated[0, -50:].tolist())


if __name__ == "__main__":
    main()
