"""
Shared training utilities for PHOTON and Baseline models.

Common functionality:
- Checkpoint saving (ZeRO-3 compatible)
- Checkpoint loading/resumption
- Weights & Biases logging
- Training loop helpers
"""

import os
from typing import Optional, Any, Dict

import torch
from accelerate import Accelerator

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    config: Any,
    step: int,
    save_dir: str,
    prefix: str = "checkpoint",
    keep_last: int = 5,
) -> Optional[str]:
    """
    Save checkpoint with ZeRO-3 compatible weight gathering.
    Automatically removes old checkpoints to keep only the last N.
    
    Args:
        accelerator: Accelerate instance
        model: The model (may be wrapped)
        config: Model config dataclass
        step: Current training step
        save_dir: Directory to save to
        prefix: Filename prefix
        keep_last: Number of recent checkpoints to keep (0 = keep all)
    
    Returns:
        Path to saved checkpoint (on main process) or None
    """
    accelerator.wait_for_everyone()
    os.makedirs(save_dir, exist_ok=True)
    
    # Gather all ZeRO-3 shards
    state_dict = accelerator.get_state_dict(model)
    
    ckpt_path = None
    if accelerator.is_main_process:
        ckpt_path = os.path.join(save_dir, f"{prefix}_{step}.pt")
        torch.save({
            "step": step,
            "model": state_dict,
            "config": config,
        }, ckpt_path)
        accelerator.print(f"[save] Checkpoint saved to {ckpt_path}")
        
        # Remove old checkpoints if keep_last > 0
        if keep_last > 0:
            _cleanup_old_checkpoints(save_dir, prefix, keep_last, accelerator)
    
    return ckpt_path


def _cleanup_old_checkpoints(
    save_dir: str,
    prefix: str,
    keep_last: int,
    accelerator: Accelerator,
):
    """Remove old checkpoints, keeping only the most recent ones."""
    import glob
    import re
    
    # Find all checkpoints with this prefix
    pattern = os.path.join(save_dir, f"{prefix}_*.pt")
    checkpoints = glob.glob(pattern)
    
    if len(checkpoints) <= keep_last:
        return
    
    # Extract step numbers and sort
    def get_step(path):
        match = re.search(rf"{prefix}_(\d+)\.pt$", path)
        return int(match.group(1)) if match else 0
    
    checkpoints_sorted = sorted(checkpoints, key=get_step, reverse=True)
    
    # Remove old ones
    to_remove = checkpoints_sorted[keep_last:]
    for old_ckpt in to_remove:
        try:
            os.remove(old_ckpt)
            accelerator.print(f"[cleanup] Removed old checkpoint: {os.path.basename(old_ckpt)}")
        except OSError as e:
            accelerator.print(f"[cleanup] Failed to remove {old_ckpt}: {e}")


def load_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    checkpoint_path: str,
    config_class: type,
) -> int:
    """
    Load checkpoint and return the step number.
    
    Args:
        accelerator: Accelerate instance
        model: The model (may be wrapped)
        checkpoint_path: Path to checkpoint file
        config_class: Config class to add to safe globals
    
    Returns:
        Step number from checkpoint
    """
    accelerator.print(f"Loading checkpoint: {checkpoint_path}")
    
    # Add config class to safe globals for PyTorch 2.6+
    torch.serialization.add_safe_globals([config_class])
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    # Load model weights
    unwrapped = accelerator.unwrap_model(model)
    state_dict = ckpt.get("model", ckpt.get("model_state_dict", {}))
    
    # Handle module. prefix from DeepSpeed
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # Check for empty ZeRO shards
    empty_count = sum(1 for v in state_dict.values() if v.numel() == 0)
    if empty_count > 10:
        accelerator.print(f"WARNING: Checkpoint has {empty_count} empty tensors (bad ZeRO-3 save)")
    
    unwrapped.load_state_dict(state_dict, strict=False)
    
    step = ckpt.get("step", 0)
    accelerator.print(f"Resumed from step {step}")
    
    return step


def get_common_args(parser, default_save_dir: str = "checkpoints"):
    """Add common training arguments to parser."""
    # Data
    parser.add_argument("--dataset", type=str, default="EleutherAI/the_pile_deduplicated")
    parser.add_argument("--tokenizer", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--batch_size", type=int, default=1)
    
    # Training
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Logging & Checkpointing
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default=default_save_dir)
    parser.add_argument("--keep_last", type=int, default=5,
                        help="Keep only the last N checkpoints (0 = keep all)")
    
    # DeepSpeed
    parser.add_argument("--ds_config", type=str, default="ds/zero3_fp16.json")
    
    # Resume
    parser.add_argument("--resume", type=str, default=None, 
                        help="Path to checkpoint to resume from")
    
    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="photon",
                        help="W&B project name")
    parser.add_argument("--wandb_run", type=str, default=None,
                        help="W&B run name (auto-generated if not specified)")
    
    return parser


def init_wandb(
    accelerator: Accelerator,
    args,
    model_name: str,
    config: Any,
    n_params: int,
) -> bool:
    """
    Initialize Weights & Biases logging.
    
    Args:
        accelerator: Accelerate instance
        args: Parsed arguments
        model_name: Name of the model (e.g., "photon", "baseline")
        config: Model config dataclass
        n_params: Number of model parameters
    
    Returns:
        True if wandb is active, False otherwise
    """
    if not args.wandb:
        return False
    
    if not WANDB_AVAILABLE:
        accelerator.print("WARNING: wandb not installed. Run: pip install wandb")
        return False
    
    if accelerator.is_main_process:
        # Build config dict for wandb
        wandb_config = {
            "model": model_name,
            "n_params": n_params,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum * accelerator.num_processes,
            "steps": args.steps,
            "dataset": args.dataset,
        }
        
        # Add model config fields
        if hasattr(config, "__dataclass_fields__"):
            for field in config.__dataclass_fields__:
                wandb_config[f"model_{field}"] = getattr(config, field)
        
        run_name = args.wandb_run or f"{model_name}-{n_params // 1_000_000}M"
        
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=wandb_config,
            resume="allow",
        )
        accelerator.print(f"[wandb] Logging to project '{args.wandb_project}', run '{run_name}'")
    
    return True


def log_wandb(
    accelerator: Accelerator,
    metrics: Dict[str, float],
    step: int,
    wandb_active: bool,
):
    """
    Log metrics to Weights & Biases.
    
    Args:
        accelerator: Accelerate instance
        metrics: Dictionary of metric names to values
        step: Current training step
        wandb_active: Whether wandb is active
    """
    if not wandb_active or not accelerator.is_main_process:
        return
    
    wandb.log(metrics, step=step)


def finish_wandb(accelerator: Accelerator, wandb_active: bool):
    """Finish wandb run."""
    if wandb_active and accelerator.is_main_process:
        wandb.finish()
