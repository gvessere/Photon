"""
Shared training utilities for PHOTON and Baseline models.

Common functionality:
- Checkpoint saving (ZeRO-3 compatible)
- Checkpoint loading/resumption
- Weights & Biases logging
- Training loop helpers
"""

import os
import re
import glob
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
        payload = {
            "step": step,
            "model": state_dict,
            "config": config,
        }
        # Store wandb metadata if available
        try:
            import wandb  # type: ignore
            if wandb.run is not None:
                payload["wandb_run_id"] = wandb.run.id
                payload["wandb_run_name"] = wandb.run.name
        except Exception:
            pass
        torch.save(payload, ckpt_path)
        accelerator.print(f"[save] Checkpoint saved to {ckpt_path}")
        
        # Remove old checkpoints if keep_last > 0
        if keep_last > 0:
            _cleanup_old_checkpoints(save_dir, prefix, keep_last, accelerator)

        # Upload to W&B and keep only the latest artifact
        try:
            import wandb  # type: ignore
            if wandb.run is not None:
                # Use per-run artifact collection to avoid collisions across concurrent runs.
                art_name = f"{prefix}-{wandb.run.id}"
                art = wandb.Artifact(name=art_name, type="checkpoint")
                art.add_file(ckpt_path, name=os.path.basename(ckpt_path))
                logged = wandb.run.log_artifact(art, aliases=["latest"])
                accelerator.print(f"[wandb] Logged artifact '{art_name}:latest'")

                # Delete older artifact versions to save space
                try:
                    api = wandb.Api()
                    atype = api.artifact_type(
                        type_name="checkpoint",
                        project=f"{wandb.run.entity}/{wandb.run.project}",
                    )
                    coll = atype.collection(art_name)
                    versions_iter = None
                    if hasattr(coll, "versions"):
                        versions_iter = coll.versions()
                    elif hasattr(coll, "artifacts"):
                        versions_iter = coll.artifacts()
                    elif hasattr(coll, "__iter__"):
                        versions_iter = coll
                    if versions_iter is not None:
                        versions = list(versions_iter)
                        if versions:
                            # Keep most recent; prefer created_at if available, else fall back to version.
                            def _art_sort_key(a):
                                created_at = getattr(a, "created_at", None)
                                if created_at is not None:
                                    return created_at
                                ver = getattr(a, "version", "")
                                if isinstance(ver, str) and ver.startswith("v") and ver[1:].isdigit():
                                    return int(ver[1:])
                                return 0
                            versions.sort(key=_art_sort_key, reverse=True)
                            for old_art in versions[1:]:
                                old_art.delete()
                                accelerator.print(f"[wandb] Deleted older artifact version {old_art.name}")
                except Exception as e:
                    accelerator.print(f"[wandb] Artifact cleanup skipped: {e}")
        except Exception as e:
            accelerator.print(f"[wandb] Artifact upload failed: {e}")
    
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


def load_checkpoint_before_prepare(
    accelerator: Accelerator,
    model: torch.nn.Module,
    checkpoint_path: str,
    config_class: type,
) -> int:
    """
    Load checkpoint BEFORE accelerator.prepare() for ZeRO-3 compatibility.
    
    With ZeRO-3, parameters are sharded after prepare(). Loading must happen
    before that, when the model still has full parameters on each rank.
    
    Args:
        accelerator: Accelerate instance
        model: The model (NOT yet prepared/wrapped)
        checkpoint_path: Path to checkpoint file
        config_class: Config class to add to safe globals
    
    Returns:
        Step number from checkpoint
    """
    accelerator.print(f"Loading checkpoint: {checkpoint_path}")
    
    # Add config class to safe globals for PyTorch 2.6+
    torch.serialization.add_safe_globals([config_class])
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    # Load model weights (model is NOT wrapped yet)
    state_dict = ckpt.get("model", ckpt.get("model_state_dict", {}))
    
    # Handle module. prefix from DeepSpeed saves
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    # Check for empty ZeRO shards (indicates bad checkpoint)
    empty_count = sum(1 for v in state_dict.values() if v.numel() == 0)
    if empty_count > 10:
        accelerator.print(f"WARNING: Checkpoint has {empty_count} empty tensors (bad ZeRO-3 save)")
        accelerator.print("This checkpoint may have been saved incorrectly. Try re-saving.")
    
    # Load directly into model (not unwrapped since not yet prepared)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        accelerator.print(f"Missing keys: {len(missing)} (may be expected for new architecture)")
    if unexpected:
        accelerator.print(f"Unexpected keys: {len(unexpected)}")
    
    step = ckpt.get("step", 0)
    accelerator.print(f"Resumed from step {step}")
    
    return step


def resolve_resume_checkpoint(
    accelerator: Accelerator,
    args,
    resume_prefix: str,
) -> Optional[str]:
    """
    Resolve a local checkpoint path from either --resume or a W&B artifact.

    Args:
        accelerator: Accelerate instance
        args: Parsed args from get_common_args()
        resume_prefix: Artifact name prefix used during checkpoint saving

    Returns:
        Local checkpoint path or None if no resume source was provided
    """
    if args.resume and args.resume_artifact_run_id:
        raise ValueError("Use only one resume source: --resume or --resume_artifact_run_id")

    if args.resume:
        return args.resume

    if not args.resume_artifact_run_id:
        return None

    if not WANDB_AVAILABLE:
        raise RuntimeError("wandb is required for artifact resume. Install with: pip install wandb")

    api = wandb.Api()
    entity = (
        args.resume_artifact_entity
        or args.wandb_entity
        or os.environ.get("WANDB_ENTITY")
        or getattr(api, "default_entity", None)
    )
    project = args.resume_artifact_project or args.wandb_project
    run_id = args.resume_artifact_run_id
    alias = args.resume_artifact_alias
    artifact_name = args.resume_artifact_name or f"{resume_prefix}-{run_id}"

    def _resolve_artifact_ref() -> str:
        # Fast path: explicit full reference.
        if entity and project:
            return f"{entity}/{project}/{artifact_name}:{alias}"

        # Try progressively less/alternate-qualified refs first.
        refs_to_try = [f"{artifact_name}:{alias}"]
        if project:
            refs_to_try.insert(0, f"{project}/{artifact_name}:{alias}")
        if entity and project:
            refs_to_try.insert(0, f"{entity}/{project}/{artifact_name}:{alias}")

        last_err = None
        for ref in refs_to_try:
            try:
                api.artifact(ref)
                return ref
            except Exception as e:
                last_err = e

        # Fallback: infer entity/project from run id under the current/default entity.
        if entity:
            try:
                projects = api.projects(entity=entity)
                for p in projects:
                    project_name = getattr(p, "name", None)
                    if not project_name:
                        continue
                    runs = api.runs(f"{entity}/{project_name}", filters={"id": run_id}, per_page=1)
                    if runs and len(runs) > 0:
                        return f"{entity}/{project_name}/{artifact_name}:{alias}"
            except Exception as e:
                last_err = e

        hint = (
            f"Could not resolve artifact for run id '{run_id}'. "
            "If auto-discovery fails, set --resume_artifact_entity and/or --resume_artifact_project."
        )
        if last_err is not None:
            raise RuntimeError(f"{hint} Last error: {last_err}") from last_err
        raise RuntimeError(hint)

    artifact_ref = _resolve_artifact_ref()

    save_root = args.save_dir or "."
    os.makedirs(save_root, exist_ok=True)
    shared_resume_path = os.path.join(save_root, ".resume_artifact_path")

    if accelerator.is_main_process:
        accelerator.print(f"[resume] Downloading W&B artifact: {artifact_ref}")
        artifact = api.artifact(artifact_ref)
        artifact_dir = artifact.download(root=os.path.join(save_root, ".wandb_artifacts"))

        if args.resume_artifact_file:
            checkpoint_path = os.path.join(artifact_dir, args.resume_artifact_file)
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint file '{args.resume_artifact_file}' not found in artifact '{artifact_ref}'"
                )
        else:
            candidates = glob.glob(os.path.join(artifact_dir, "*.pt"))
            if not candidates:
                raise FileNotFoundError(f"No .pt checkpoint files found in artifact '{artifact_ref}'")

            def _extract_step(path: str) -> int:
                match = re.search(r"_(\d+)\.pt$", os.path.basename(path))
                return int(match.group(1)) if match else -1

            checkpoint_path = sorted(candidates, key=lambda p: (_extract_step(p), os.path.getmtime(p)))[-1]

        with open(shared_resume_path, "w", encoding="utf-8") as f:
            f.write(checkpoint_path)
        accelerator.print(f"[resume] Using checkpoint from artifact: {checkpoint_path}")

    accelerator.wait_for_everyone()
    with open(shared_resume_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def get_common_args(parser, default_save_dir: str = "checkpoints"):
    """Add common training arguments to parser."""
    # Data
    parser.add_argument("--dataset", type=str, default="EleutherAI/the_pile_deduplicated")
    parser.add_argument("--tokenizer", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--eval_split", type=str, default="validation",
                        help="Dataset split for evaluation (if missing, eval is derived from train)")
    parser.add_argument("--eval_from_train_examples", type=int, default=10000,
                        help="When eval split is missing, reserve this many train examples for eval")
    parser.add_argument("--batch_size", type=int, default=3)
    
    # Training
    parser.add_argument("--steps", type=int, default=100000)
    # With 2 processes and batch_size=3, grad_accum=43 -> ~258 effective (≈256 target)
    parser.add_argument("--grad_accum", type=int, default=43)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Logging & Checkpointing
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--save_dir", type=str, default=default_save_dir)
    parser.add_argument("--keep_last", type=int, default=5,
                        help="Keep only the last N checkpoints (0 = keep all)")
    
    # DeepSpeed
    parser.add_argument("--ds_config", type=str, default="ds/zero3_fp16.json")
    
    # Resume
    parser.add_argument("--resume", type=str, default=None, 
                        help="Path to checkpoint to resume from")
    parser.add_argument("--resume_artifact_run_id", type=str, default=None,
                        help="W&B run id to resume from saved checkpoint artifact")
    parser.add_argument("--resume_artifact_name", type=str, default=None,
                        help="W&B artifact name (default: <model_prefix>-<resume_artifact_run_id>)")
    parser.add_argument("--resume_artifact_alias", type=str, default="latest",
                        help="W&B artifact alias/version (default: latest)")
    parser.add_argument("--resume_artifact_file", type=str, default=None,
                        help="Checkpoint file path inside artifact (default: auto-select .pt)")
    parser.add_argument("--resume_artifact_project", type=str, default=None,
                        help="W&B project for artifact resume (default: --wandb_project)")
    parser.add_argument("--resume_artifact_entity", type=str, default=None,
                        help="W&B entity for artifact resume (default: auto-detect)")
    
    # Weights & Biases
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="photon",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity/team")
    parser.add_argument("--wandb_run", type=str, default=None,
                        help="W&B run name (auto-generated if not specified)")
    parser.add_argument("--wandb_id", type=str, default=None,
                        help="W&B run id for resuming the same run (sets wandb.init id=..., resume='allow')")
    
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
        run_id = args.wandb_id
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            id=run_id,
            config=wandb_config,
            resume="allow",
        )
        accelerator.print(f"[wandb] Logging to project '{args.wandb_project}', run '{run_name}'"
                          f"{' (resume id=' + run_id + ')' if run_id else ''}")
    
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
