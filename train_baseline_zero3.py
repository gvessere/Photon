#!/usr/bin/env python
"""
Baseline Transformer Training with Accelerate + DeepSpeed ZeRO-3

Launch with:
    accelerate launch --num_processes 2 train_baseline_zero3.py

Resume from checkpoint:
    accelerate launch --num_processes 2 train_baseline_zero3.py --resume checkpoints_baseline/checkpoint_1000.pt
    accelerate launch --num_processes 2 train_baseline_zero3.py --resume_artifact_run_id <run_id>

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
from train_utils import (
    save_checkpoint, load_checkpoint_before_prepare, resolve_resume_checkpoint, get_common_args,
    init_wandb, log_wandb, finish_wandb, capture_data_state, restore_data_state
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Baseline Transformer with DeepSpeed")
    
    # Add common args with baseline-specific save dir
    get_common_args(parser, default_save_dir="checkpoints_baseline")
    
    # Baseline-specific args (paper-aligned vanilla 600M defaults)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--d_model", type=int, default=1664)
    parser.add_argument("--n_heads", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=16)
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
        tie_embeddings=False,
    )
    
    # Create model
    with accelerator.main_process_first():
        accelerator.print("Creating model...")
        model = BaselineLM(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        accelerator.print(f"Model parameters: {n_params / 1e6:.2f}M")
    
    # Initialize wandb
    wandb_active = init_wandb(accelerator, args, "baseline", cfg, n_params)
    
    # Resume from checkpoint if specified (BEFORE prepare for ZeRO-3)
    start_step = 0
    data_state = None
    resume_path = resolve_resume_checkpoint(accelerator, args, resume_prefix="baseline")
    if resume_path:
        start_step, data_state = load_checkpoint_before_prepare(
            accelerator, model, resume_path, BaselineConfig
        )

    # Load dataset
    with accelerator.main_process_first():
        accelerator.print("Loading dataset...")
        train_loader, eval_loader, tokenizer = create_dataloaders(
            dataset_name=args.dataset,
            tokenizer_name=args.tokenizer,
            block_size=args.max_seq_len,
            batch_size=args.batch_size,
            streaming=True,
            eval_split=args.eval_split if args.eval_split else None,
            eval_from_train_examples=args.eval_from_train_examples,
        )
        cfg.eos_token_id = tokenizer.eos_token_id
        cfg.pad_token_id = tokenizer.pad_token_id
        cfg.vocab_size = len(tokenizer)
    
    # Prepare model and dataloaders
    if eval_loader is not None:
        model, train_loader, eval_loader = accelerator.prepare(model, train_loader, eval_loader)
    else:
        model, train_loader = accelerator.prepare(model, train_loader)

    # Restore loader cursor state after prepare() when supported.
    restore_data_state(accelerator, train_loader, data_state)
    
    accelerator.print("Starting training...")
    
    # Training loop
    model.train()
    it = iter(train_loader)
    running_loss = 0.0
    # Per-rank cumulative tokens; reduced across ranks when logging.
    tokens_seen_local = start_step * args.batch_size * args.max_seq_len
    
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
        tokens_seen_local += batch["labels"].numel()
        
        # Logging
        if step % args.log_every == 0:
            optimizer_step = step // args.grad_accum
            tokens_seen_global = int(
                accelerator.reduce(
                    torch.tensor(tokens_seen_local, device=accelerator.device, dtype=torch.long),
                    reduction="sum",
                ).item()
            )
        if accelerator.is_main_process and step % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            accelerator.print(
                f"step {step:6d} | opt {optimizer_step:6d} | tok {tokens_seen_global:,} | loss {avg_loss:.4f}"
            )
            
            # Log to wandb
            log_wandb(
                accelerator,
                {
                    "train/loss_lm": avg_loss,
                    "train/optimizer_step": optimizer_step,
                    "train/tokens_seen": tokens_seen_global,
                },
                step,
                wandb_active,
            )
            
            running_loss = 0.0

        # Evaluation
        if eval_loader is not None and step % args.eval_every == 0:
            model.eval()
            total_loss, total_tokens = 0.0, 0
            optimizer_step = step // args.grad_accum
            tokens_seen_global = int(
                accelerator.reduce(
                    torch.tensor(tokens_seen_local, device=accelerator.device, dtype=torch.long),
                    reduction="sum",
                ).item()
            )

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
                    "eval/optimizer_step": optimizer_step,
                    "eval/tokens_seen": tokens_seen_global,
                }, step, wandb_active)

            model.train()
        
        # Checkpointing
        if args.save_dir and step % args.save_every == 0:
            data_state = capture_data_state(accelerator, train_loader)
            save_checkpoint(
                accelerator=accelerator,
                model=model,
                config=cfg,
                step=step,
                save_dir=args.save_dir,
                prefix="baseline",
                keep_last=args.keep_last,
                data_state=data_state,
            )
    
    # Finish wandb
    finish_wandb(accelerator, wandb_active)
    
    accelerator.print("Training complete!")


if __name__ == "__main__":
    main()
