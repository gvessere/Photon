#!/usr/bin/env python
"""
Baseline Transformer Training with Accelerate + DeepSpeed ZeRO-3

Launch with:
    accelerate launch --num_processes 2 train_baseline_zero3.py

A baseline for comparison with PHOTON.
"""

import os
import math
import argparse

import torch
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin

from baseline import BaselineConfig, BaselineLM
from photon.data import create_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train Baseline Transformer with DeepSpeed")
    
    # Data
    parser.add_argument("--dataset", type=str, default="EleutherAI/the_pile_deduplicated")
    parser.add_argument("--tokenizer", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    
    # Training
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="checkpoints_baseline")
    
    # DeepSpeed
    parser.add_argument("--ds_config", type=str, default="ds/zero3_fp16.json")
    
    # Model - sized to match PHOTON ~650M params
    parser.add_argument("--d_model", type=int, default=1536)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_layers", type=int, default=24)
    parser.add_argument("--d_ff", type=int, default=4096)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
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
    accelerator.print("=" * 60)
    
    # Config
    cfg = BaselineConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        gradient_checkpointing=True,
    )
    
    # Model
    with accelerator.main_process_first():
        accelerator.print("Creating model...")
        model = BaselineLM(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        accelerator.print(f"Model parameters: {n_params / 1e6:.2f}M")
    
    # Data
    with accelerator.main_process_first():
        accelerator.print("Loading dataset...")
        train_loader, _, tokenizer = create_dataloaders(
            dataset_name=args.dataset,
            tokenizer_name=args.tokenizer,
            block_size=args.max_seq_len,
            batch_size=args.batch_size,
            streaming=True,
        )
        cfg.eos_token_id = tokenizer.eos_token_id
        cfg.pad_token_id = tokenizer.pad_token_id
        cfg.vocab_size = len(tokenizer)
    
    # Prepare
    model, train_loader = accelerator.prepare(model, train_loader)
    
    accelerator.print("Starting training...")
    model.train()
    it = iter(train_loader)
    running_loss = 0.0
    
    for step in range(1, args.steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        
        with accelerator.accumulate(model):
            loss = model(**batch)["loss"]
            accelerator.backward(loss)
        
        running_loss += loss.item()
        
        if accelerator.is_main_process and step % args.log_every == 0:
            accelerator.print(f"step {step:6d} | loss {running_loss / args.log_every:.4f}")
            running_loss = 0.0
        
        if args.save_dir and step % args.save_every == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                os.makedirs(args.save_dir, exist_ok=True)
                unwrapped = accelerator.unwrap_model(model)
                accelerator.save({
                    "step": step,
                    "model": unwrapped.state_dict(),
                    "config": cfg,
                }, f"{args.save_dir}/checkpoint_{step}.pt")
    
    accelerator.print("Training complete!")


if __name__ == "__main__":
    main()
