"""
Shared training utilities for PHOTON and Baseline models.

Common functionality:
- Checkpoint saving (ZeRO-3 compatible)
- Checkpoint loading/resumption
- Training loop helpers
"""

import os
from typing import Optional, Any, Dict

import torch
from accelerate import Accelerator


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    config: Any,
    step: int,
    save_dir: str,
    prefix: str = "checkpoint"
) -> Optional[str]:
    """
    Save checkpoint with ZeRO-3 compatible weight gathering.
    
    Args:
        accelerator: Accelerate instance
        model: The model (may be wrapped)
        config: Model config dataclass
        step: Current training step
        save_dir: Directory to save to
        prefix: Filename prefix
    
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
    
    return ckpt_path


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
    
    # DeepSpeed
    parser.add_argument("--ds_config", type=str, default="ds/zero3_fp16.json")
    
    # Resume
    parser.add_argument("--resume", type=str, default=None, 
                        help="Path to checkpoint to resume from")
    
    return parser
