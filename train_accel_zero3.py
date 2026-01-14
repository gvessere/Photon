#!/usr/bin/env python
"""
PHOTON Training Script with Accelerate + DeepSpeed ZeRO-3

Launch with:
    accelerate launch --num_processes 2 train_accel_zero3.py

Or with config:
    accelerate launch --config_file accelerate_config.yaml train_accel_zero3.py

This script enables multi-GPU training on 2×T4 GPUs with:
- ZeRO-3 model sharding (fits large models)
- fp16 mixed precision (T4 compatible, no bf16)
- Gradient accumulation
- Periodic evaluation and checkpointing
"""

import os
import math
import argparse
from typing import Optional

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin

# Import PHOTON modules
from photon import PhotonConfig, PhotonLM
from photon.data import create_dataloaders, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Train PHOTON with Accelerate + DeepSpeed")
    
    # Data
    parser.add_argument("--dataset", type=str, default="EleutherAI/the_pile_deduplicated")
    parser.add_argument("--tokenizer", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    
    # Training
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Logging
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    
    # DeepSpeed config
    parser.add_argument("--ds_config", type=str, default="ds/zero3_fp16.json")
    
    # Model config
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=5120)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize Accelerate with DeepSpeed ZeRO-3
    ds_plugin = DeepSpeedPlugin(
        zero_stage=3,
        hf_ds_config=args.ds_config
    )
    accelerator = Accelerator(
        mixed_precision="fp16",  # T4 compatible (not bf16)
        deepspeed_plugin=ds_plugin,
        gradient_accumulation_steps=args.grad_accum,
    )
    
    accelerator.print("=" * 60)
    accelerator.print("PHOTON Training with Accelerate + DeepSpeed ZeRO-3")
    accelerator.print(f"  Processes: {accelerator.num_processes}")
    accelerator.print(f"  Mixed precision: {accelerator.mixed_precision}")
    accelerator.print(f"  Device: {accelerator.device}")
    accelerator.print("=" * 60)
    
    # Create model config
    cfg = PhotonConfig(
        n_layers_enc=args.n_layers,
        n_layers_dec=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    
    # Create model
    with accelerator.main_process_first():
        accelerator.print("Creating model...")
        model = PhotonLM(cfg)
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        accelerator.print(f"Model parameters: {n_params / 1e6:.2f}M")
    
    # Create dataloaders
    with accelerator.main_process_first():
        accelerator.print("Loading dataset...")
        train_loader, eval_loader, tokenizer = create_dataloaders(
            dataset_name=args.dataset,
            tokenizer_name=args.tokenizer,
            block_size=args.block_size,
            batch_size=args.batch_size,
            streaming=True,
        )
        
        # Update config with tokenizer info
        cfg.eos_token_id = tokenizer.eos_token_id
        cfg.pad_token_id = tokenizer.pad_token_id
        cfg.vocab_size = len(tokenizer)
    
    # Create optimizer (DeepSpeed will manage this with ZeRO-3)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )
    
    # Prepare with Accelerate
    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )
    
    accelerator.print("Starting training...")
    
    # Training loop
    model.train()
    it = iter(train_loader)
    running_loss = 0.0
    
    for step in range(1, args.steps + 1):
        # Get batch
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        
        # Forward pass with autocast
        with accelerator.autocast():
            out = model(**batch)
            loss = out["loss"]
        
        # Backward pass
        accelerator.backward(loss)
        
        # Gradient accumulation
        if step % args.grad_accum == 0:
            if args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        running_loss += loss.item()
        
        # Logging
        if accelerator.is_main_process and step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            accelerator.print(f"step {step:6d} | loss {avg_loss:.4f}")
            running_loss = 0.0
        
        # Evaluation
        if eval_loader is not None and step % args.eval_every == 0:
            model.eval()
            total_loss, total_tokens = 0.0, 0
            
            with torch.no_grad():
                for i, eval_batch in enumerate(eval_loader):
                    if i >= 100:
                        break
                    with accelerator.autocast():
                        out = model(**eval_batch)
                    total_loss += out["loss"].item() * eval_batch["labels"].numel()
                    total_tokens += eval_batch["labels"].numel()
            
            if total_tokens > 0:
                mean_loss = total_loss / total_tokens
                ppl = math.exp(min(mean_loss, 100))
                
                if accelerator.is_main_process:
                    accelerator.print(f"[eval] step {step} | loss {mean_loss:.4f} | ppl {ppl:.2f}")
            
            model.train()
        
        # Checkpointing
        if args.save_dir and step % args.save_every == 0:
            accelerator.wait_for_everyone()
            
            if accelerator.is_main_process:
                os.makedirs(args.save_dir, exist_ok=True)
                
                # Get unwrapped model for saving
                unwrapped_model = accelerator.unwrap_model(model)
                
                # Save checkpoint
                ckpt_path = os.path.join(args.save_dir, f"checkpoint_{step}.pt")
                accelerator.save({
                    "step": step,
                    "model_state_dict": unwrapped_model.state_dict(),
                    "config": cfg,
                }, ckpt_path)
                
                accelerator.print(f"[save] Checkpoint saved to {ckpt_path}")
    
    accelerator.print("Training complete!")


if __name__ == "__main__":
    main()
