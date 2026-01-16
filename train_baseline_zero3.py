#!/usr/bin/env python
"""
Baseline Transformer Training with Accelerate + DeepSpeed ZeRO-3

Launch with:
    accelerate launch --num_processes 2 train_baseline_zero3.py

Resume from checkpoint:
    accelerate launch --num_processes 2 train_baseline_zero3.py --resume checkpoints_baseline/checkpoint_1000.pt

A baseline for comparison with PHOTON.
"""

import os
import argparse

import torch
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin

from baseline import BaselineConfig, BaselineLM
from photon.data import create_dataloaders
from train_utils import (
    save_checkpoint, load_checkpoint, get_common_args,
    init_wandb, log_wandb, finish_wandb
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Baseline Transformer with DeepSpeed")
    
    # Add common args with baseline-specific save dir
    get_common_args(parser, default_save_dir="checkpoints_baseline")
    
    # Baseline-specific args
    parser.add_argument("--max_seq_len", type=int, default=2048)
    
    # Model - sized to match PHOTON ~650M params
    parser.add_argument("--d_model", type=int, default=1536)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_layers", type=int, default=24)
    parser.add_argument("--d_ff", type=int, default=4096)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize Accelerate with DeepSpeed ZeRO-3
    ds_plugin = DeepSpeedPlugin(zero_stage=3, hf_ds_config=args.ds_config)
    accelerator = Accelerator(
        mixed_precision="fp16",
        deepspeed_plugin=ds_plugin,
        gradient_accumulation_steps=args.grad_accum,
    )
    
    accelerator.print("=" * 60)
    accelerator.print("Baseline Transformer Training with DeepSpeed ZeRO-3")
    accelerator.print(f"  Processes: {accelerator.num_processes}")
    accelerator.print(f"  Mixed precision: {accelerator.mixed_precision}")
    accelerator.print(f"  Batch size: {args.batch_size} x {args.grad_accum} x {accelerator.num_processes}")
    accelerator.print("=" * 60)
    
    # Model config
    cfg = BaselineConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    
    # Create model
    with accelerator.main_process_first():
        accelerator.print("Creating model...")
        model = BaselineLM(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        accelerator.print(f"Model parameters: {n_params / 1e6:.2f}M")
    
    # Initialize wandb
    wandb_active = init_wandb(accelerator, args, "baseline", cfg, n_params)
    
    # Load dataset
    with accelerator.main_process_first():
        accelerator.print("Loading dataset...")
        train_loader, eval_loader, tokenizer = create_dataloaders(
            dataset_name=args.dataset,
            tokenizer_name=args.tokenizer,
            block_size=args.max_seq_len,
            batch_size=args.batch_size,
            streaming=True,
        )
        cfg.eos_token_id = tokenizer.eos_token_id
        cfg.pad_token_id = tokenizer.pad_token_id
        cfg.vocab_size = len(tokenizer)
    
    # Prepare model and dataloader
    model, train_loader = accelerator.prepare(model, train_loader)
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(accelerator, model, args.resume, BaselineConfig)
    
    accelerator.print("Starting training...")
    
    # Training loop
    model.train()
    it = iter(train_loader)
    running_loss = 0.0
    
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
        
        # Logging
        if accelerator.is_main_process and step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            accelerator.print(f"step {step:6d} | loss {avg_loss:.4f}")
            
            # Log to wandb
            log_wandb(accelerator, {"train/loss_lm": avg_loss}, step, wandb_active)
            
            running_loss = 0.0
        
        # Checkpointing
        if args.save_dir and step % args.save_every == 0:
            save_checkpoint(
                accelerator=accelerator,
                model=model,
                config=cfg,
                step=step,
                save_dir=args.save_dir,
                prefix="baseline",
                keep_last=args.keep_last,
            )
    
    # Finish wandb
    finish_wandb(accelerator, wandb_active)
    
    accelerator.print("Training complete!")


if __name__ == "__main__":
    main()
