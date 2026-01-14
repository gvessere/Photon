"""
PHOTON Training Module

Supports:
- Single GPU training with AMP (fp16)
- Multi-GPU with Accelerate + DeepSpeed ZeRO-3
- Gradient accumulation
- Evaluation and checkpointing
"""

import math
import os
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from .config import PhotonConfig
from .model import PhotonLM


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 200,
    use_amp: bool = True,
) -> tuple:
    """
    Evaluate model on a dataloader.
    
    Args:
        model: PhotonLM model
        loader: Evaluation dataloader
        device: Device to run on
        max_batches: Maximum batches to evaluate
        use_amp: Use automatic mixed precision
    
    Returns:
        (mean_loss, perplexity)
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            
            batch = {k: v.to(device) for k, v in batch.items()}
            
            if use_amp:
                with autocast("cuda", dtype=torch.float16):
                    out = model(**batch)
            else:
                out = model(**batch)
            
            loss = out["loss"] if isinstance(out, dict) else out.loss
            tokens = batch["labels"].numel()
            total_loss += loss.item() * tokens
            total_tokens += tokens
    
    mean_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(mean_loss, 100))  # Cap to avoid overflow
    model.train()
    
    return mean_loss, ppl


def train_single_gpu(
    model: PhotonLM,
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
    Single-GPU training loop with AMP (fp16).
    
    This is T4-compatible: uses fp16 (not bf16).
    
    Args:
        model: PhotonLM model
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
        max_grad_norm: Maximum gradient norm for clipping
        device: Device string
    
    Returns:
        Dict with training history
    """
    device = torch.device(device)
    model.to(device)
    model.train()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay
    )
    
    # GradScaler for fp16 - explicitly specify "cuda" for T4 compatibility
    scaler = GradScaler("cuda")
    
    # Training loop
    history = {"train_loss": [], "eval_loss": [], "eval_ppl": []}
    it = iter(train_loader)
    accumulated_loss = 0.0
    
    for step in range(1, steps + 1):
        # Get batch
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        
        # Forward with AMP - use explicit fp16 dtype for T4 compatibility
        with autocast("cuda", dtype=torch.float16):
            out = model(**batch)
            loss = out["loss"] if isinstance(out, dict) else out.loss
            loss = loss / grad_accum_steps
        
        # Backward
        scaler.scale(loss).backward()
        accumulated_loss += loss.item()
        
        # Optimizer step (with gradient accumulation)
        if step % grad_accum_steps == 0:
            # Unscale for gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            # Log
            if (step // grad_accum_steps) % log_every == 0:
                avg_loss = accumulated_loss * grad_accum_steps / log_every
                print(f"step {step} loss {avg_loss:.4f}")
                history["train_loss"].append(avg_loss)
                accumulated_loss = 0.0
        
        # Evaluate
        if eval_loader is not None and step % eval_every == 0:
            val_loss, val_ppl = evaluate(model, eval_loader, device)
            print(f"[eval] step {step} loss {val_loss:.4f} ppl {val_ppl:.2f}")
            history["eval_loss"].append(val_loss)
            history["eval_ppl"].append(val_ppl)
        
        # Save checkpoint
        if save_dir is not None and step % save_every == 0:
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = os.path.join(save_dir, f"checkpoint_{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": model.cfg,
            }, ckpt_path)
            print(f"[save] checkpoint saved to {ckpt_path}")
    
    return history


def create_accelerate_training_fn():
    """
    Create a training function compatible with Accelerate.
    
    This is meant to be used inside train_accel_zero3.py script.
    """
    from accelerate import Accelerator
    from accelerate.utils import DeepSpeedPlugin
    
    def train_with_accelerate(
        model: PhotonLM,
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
        ds_config_path: str = "ds/zero3_fp16.json",
    ) -> Dict[str, Any]:
        """
        Multi-GPU training with Accelerate + DeepSpeed ZeRO-3.
        
        Args:
            model: PhotonLM model
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
            ds_config_path: Path to DeepSpeed config JSON
        
        Returns:
            Dict with training history
        """
        # Initialize Accelerate with DeepSpeed
        ds_plugin = DeepSpeedPlugin(
            zero_stage=3,
            hf_ds_config=ds_config_path
        )
        accelerator = Accelerator(
            mixed_precision="fp16",
            deepspeed_plugin=ds_plugin,
            gradient_accumulation_steps=grad_accum_steps,
        )
        
        # Optimizer (DeepSpeed will manage this)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=weight_decay
        )
        
        # Prepare with Accelerate
        model, optimizer, train_loader = accelerator.prepare(
            model, optimizer, train_loader
        )
        if eval_loader is not None:
            eval_loader = accelerator.prepare(eval_loader)
        
        # Training loop
        history = {"train_loss": [], "eval_loss": [], "eval_ppl": []}
        model.train()
        it = iter(train_loader)
        
        for step in range(1, steps + 1):
            # Get batch
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            
            # Forward with automatic mixed precision
            with accelerator.autocast():
                out = model(**batch)
                loss = out["loss"] if isinstance(out, dict) else out.loss
            
            # Backward
            accelerator.backward(loss)
            
            # Optimizer step
            if step % grad_accum_steps == 0:
                if max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            # Log (main process only)
            if accelerator.is_main_process and step % log_every == 0:
                accelerator.print(f"step {step} loss {loss.item():.4f}")
                history["train_loss"].append(loss.item())
            
            # Evaluate
            if eval_loader is not None and step % eval_every == 0:
                model.eval()
                total_loss, total_tokens = 0.0, 0
                with torch.no_grad():
                    for i, eval_batch in enumerate(eval_loader):
                        if i >= 200:
                            break
                        with accelerator.autocast():
                            out = model(**eval_batch)
                        eval_loss = out["loss"] if isinstance(out, dict) else out.loss
                        total_loss += eval_loss.item() * eval_batch["labels"].numel()
                        total_tokens += eval_batch["labels"].numel()
                
                mean_loss = total_loss / max(total_tokens, 1)
                ppl = math.exp(min(mean_loss, 100))
                
                if accelerator.is_main_process:
                    accelerator.print(f"[eval] step {step} loss {mean_loss:.4f} ppl {ppl:.2f}")
                    history["eval_loss"].append(mean_loss)
                    history["eval_ppl"].append(ppl)
                
                model.train()
            
            # Save checkpoint
            if save_dir is not None and step % save_every == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    os.makedirs(save_dir, exist_ok=True)
                    # Save unwrapped model
                    unwrapped = accelerator.unwrap_model(model)
                    accelerator.save({
                        "step": step,
                        "model_state_dict": unwrapped.state_dict(),
                        "config": unwrapped.cfg,
                    }, os.path.join(save_dir, f"checkpoint_{step}.pt"))
                    accelerator.print(f"[save] checkpoint saved at step {step}")
        
        return history
    
    return train_with_accelerate
