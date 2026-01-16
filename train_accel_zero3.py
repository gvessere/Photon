#!/usr/bin/env python
"""
PHOTON Training Script with Accelerate + DeepSpeed ZeRO-3

Launch with:
    accelerate launch --num_processes 2 train_accel_zero3.py

Resume from checkpoint:
    accelerate launch --num_processes 2 train_accel_zero3.py --resume checkpoints_photon/checkpoint_1000.pt

This script enables multi-GPU training on 2×T4 GPUs with:
- ZeRO-3 model sharding (fits large models)
- fp16 mixed precision (T4 compatible, no bf16)
- Gradient accumulation
- Periodic evaluation and checkpointing
"""

import os
import math
import argparse

import torch
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin

# Import PHOTON modules
from photon import PhotonConfig, PhotonLM
from photon.data import create_dataloaders
from train_utils import (
    save_checkpoint, load_checkpoint_before_prepare, get_common_args,
    init_wandb, log_wandb, finish_wandb
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PHOTON with Accelerate + DeepSpeed")
    
    # Add common args with photon-specific save dir
    get_common_args(parser, default_save_dir="checkpoints_photon")
    
    # PHOTON-specific args
    parser.add_argument("--block_size", type=int, default=2048)
    
    # Model config - defaults sized for 2×T4 (15GB each)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    
    # Loss weighting (Paper Eq. 7)
    parser.add_argument("--lambda_lm", type=float, default=1.0, help="Weight for LM loss")
    parser.add_argument("--lambda_ctx", type=float, default=1.0, help="Weight for next-context loss")
    parser.add_argument("--lambda_rec", type=float, default=1.0, help="Weight for reconstruction loss")
    
    # Conditioning detach (True = prevent collusion, False = joint encoder-decoder learning)
    parser.add_argument("--no_detach_conditioning", action="store_true",
                       help="Don't detach conditioning paths (allow encoder-decoder gradient flow)")
    
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
    accelerator.print(f"  Batch size: {args.batch_size} x {args.grad_accum} x {accelerator.num_processes}")
    accelerator.print("=" * 60)
    
    # Create model config
    cfg = PhotonConfig(
        n_layers_enc=args.n_layers,
        n_layers_dec=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        gradient_checkpointing=args.gradient_checkpointing,
        # Loss weights (Paper Eq. 7)
        lambda_lm=args.lambda_lm,
        lambda_ctx=args.lambda_ctx,
        lambda_rec=args.lambda_rec,
        # Conditioning detach behavior
        detach_conditioning=not args.no_detach_conditioning,
    )
    
    # Create model
    with accelerator.main_process_first():
        accelerator.print("Creating model...")
        model = PhotonLM(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        accelerator.print(f"Model parameters: {n_params / 1e6:.2f}M")
    
    # Initialize wandb
    wandb_active = init_wandb(accelerator, args, "photon", cfg, n_params)
    
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
        cfg.eos_token_id = tokenizer.eos_token_id
        cfg.pad_token_id = tokenizer.pad_token_id
        cfg.vocab_size = len(tokenizer)
    
    # Resume from checkpoint if specified (BEFORE prepare for ZeRO-3)
    start_step = 0
    if args.resume:
        start_step = load_checkpoint_before_prepare(accelerator, model, args.resume, PhotonConfig)
    
    # Prepare model and dataloader
    model, train_loader = accelerator.prepare(model, train_loader)
    
    accelerator.print("Starting training...")
    
    # Training loop
    model.train()
    it = iter(train_loader)
    running_loss = 0.0
    running_loss_rec = 0.0
    running_loss_ctx = 0.0
    running_loss_lm = 0.0
    
    for step in range(start_step + 1, args.steps + 1):
        # Get batch
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        
        # Forward and backward
        with accelerator.accumulate(model):
            out = model(**batch)
            loss = out["loss"]
            accelerator.backward(loss)
        
        running_loss += loss.item()
        running_loss_rec += out.get("loss_rec", torch.tensor(0.0)).item()
        running_loss_ctx += out.get("loss_ctx", torch.tensor(0.0)).item()
        running_loss_lm += out.get("loss_lm", torch.tensor(0.0)).item()
        
        # Logging
        if accelerator.is_main_process and step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            avg_rec = running_loss_rec / args.log_every
            avg_ctx = running_loss_ctx / args.log_every
            avg_lm = running_loss_lm / args.log_every
            accelerator.print(f"step {step:6d} | loss {avg_loss:.4f} | rec {avg_rec:.4f} | ctx {avg_ctx:.4f} | lm {avg_lm:.4f}")
            
            # Log to wandb
            log_wandb(accelerator, {
                "train/loss": avg_loss,
                "train/loss_rec": avg_rec,
                "train/loss_ctx": avg_ctx,
                "train/loss_lm": avg_lm,
            }, step, wandb_active)
            
            running_loss = 0.0
            running_loss_rec = 0.0
            running_loss_ctx = 0.0
            running_loss_lm = 0.0
        
        # Evaluation
        if eval_loader is not None and step % args.eval_every == 0:
            model.eval()
            total_loss, total_tokens = 0.0, 0
            
            with torch.no_grad():
                for i, eval_batch in enumerate(eval_loader):
                    if i >= 100:
                        break
                    out = model(**eval_batch)
                    total_loss += out["loss"].item() * eval_batch["labels"].numel()
                    total_tokens += eval_batch["labels"].numel()
            
            if total_tokens > 0:
                mean_loss = total_loss / total_tokens
                ppl = math.exp(min(mean_loss, 100))
                accelerator.print(f"[eval] step {step} | loss {mean_loss:.4f} | ppl {ppl:.2f}")
                
                # Log to wandb
                log_wandb(accelerator, {
                    "eval/loss": mean_loss,
                    "eval/ppl": ppl,
                }, step, wandb_active)
            
            model.train()
        
        # Checkpointing
        if args.save_dir and step % args.save_every == 0:
            save_checkpoint(
                accelerator=accelerator,
                model=model,
                config=cfg,
                step=step,
                save_dir=args.save_dir,
                prefix="photon",
                keep_last=args.keep_last,
            )
    
    # Finish wandb
    finish_wandb(accelerator, wandb_active)
    
    accelerator.print("Training complete!")


if __name__ == "__main__":
    main()
