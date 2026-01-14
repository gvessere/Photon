"""
Baseline Transformer Training

Single-GPU training with AMP for comparison with PHOTON.
"""

import math
import os
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from .model import BaselineConfig, BaselineLM


def train_baseline(
    model: BaselineLM,
    train_loader: DataLoader,
    eval_loader: Optional[DataLoader] = None,
    steps: int = 10_000,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    log_every: int = 50,
    eval_every: int = 500,
    save_every: int = 1000,
    save_dir: Optional[str] = None,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Train baseline transformer with AMP (fp16).
    
    Args:
        model: BaselineLM model
        train_loader: Training dataloader
        eval_loader: Optional evaluation dataloader
        steps: Total training steps
        lr: Learning rate
        weight_decay: AdamW weight decay
        log_every: Log every N steps
        eval_every: Evaluate every N steps
        save_every: Save checkpoint every N steps
        save_dir: Directory for checkpoints
        grad_accum_steps: Gradient accumulation steps
        max_grad_norm: Maximum gradient norm
        device: Device string
    
    Returns:
        Training history dict
    """
    device = torch.device(device)
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay
    )
    
    scaler = GradScaler("cuda")
    
    history = {"train_loss": [], "eval_loss": [], "eval_ppl": []}
    it = iter(train_loader)
    accumulated_loss = 0.0
    
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        
        with autocast("cuda", dtype=torch.float16):
            out = model(**batch)
            loss = out["loss"] / grad_accum_steps
        
        scaler.scale(loss).backward()
        accumulated_loss += loss.item()
        
        if step % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            if (step // grad_accum_steps) % log_every == 0:
                avg_loss = accumulated_loss * grad_accum_steps / log_every
                print(f"step {step} loss {avg_loss:.4f}")
                history["train_loss"].append(avg_loss)
                accumulated_loss = 0.0
        
        if eval_loader is not None and step % eval_every == 0:
            val_loss, val_ppl = evaluate(model, eval_loader, device)
            print(f"[eval] step {step} loss {val_loss:.4f} ppl {val_ppl:.2f}")
            history["eval_loss"].append(val_loss)
            history["eval_ppl"].append(val_ppl)
        
        if save_dir is not None and step % save_every == 0:
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = os.path.join(save_dir, f"baseline_checkpoint_{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": model.cfg,
            }, ckpt_path)
            print(f"[save] checkpoint saved to {ckpt_path}")
    
    return history


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 200,
) -> tuple:
    """Evaluate model."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast("cuda", dtype=torch.float16):
                out = model(**batch)
            loss = out["loss"]
            tokens = batch["labels"].numel()
            total_loss += loss.item() * tokens
            total_tokens += tokens
    
    mean_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(mean_loss, 100))
    model.train()
    return mean_loss, ppl
