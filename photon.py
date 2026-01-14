"""
PHOTON: Hierarchical Latent Language Model

This file provides backward compatibility by importing from the modular package.
For new code, import directly from the photon package:

    from photon import PhotonConfig, PhotonLM
    from photon.data import create_dataloaders
    from photon.inference import generate_photon

For training on 2×T4 GPUs with Accelerate + DeepSpeed ZeRO-3:
    
    accelerate launch --num_processes 2 train_accel_zero3.py
"""

# Re-export all public APIs for backward compatibility
from photon import PhotonConfig, PhotonLM, create_dataloaders, collate_fn, generate_photon
from photon.model import (
    CausalSelfAttention,
    MLP,
    TransformerBlock,
    CtxTransformer,
    ConcatChunker,
    LinearChunker,
    TableMatchedConverter,
    LatentARHead,
    GaussianLatentLoss,
    RotaryEmbedding,
)
from photon.train import train_single_gpu, evaluate
from photon.data import PhotonDataset


# =============================================================================
# Example usage / Quick start
# =============================================================================

def main():
    """Example training script using the modular PHOTON implementation."""
    import torch
    from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

    # Create config
    cfg = PhotonConfig(
        vocab_size=len(tokenizer),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        # Use smaller model for single GPU
        n_layers_enc=3,
        n_layers_dec=3,
        n_heads=8,
        d_ff=2048,
        gradient_checkpointing=True,  # Save memory
    )
    
    print(f"Block size: {cfg.block_size}")
    print(f"d_latent: {cfg.d_latent}")
    
    # Create model
    model = PhotonLM(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")
    
    # Create dataloaders
    train_loader, _, _ = create_dataloaders(
        dataset_name="EleutherAI/the_pile_deduplicated",
        tokenizer_name="mistralai/Mistral-7B-v0.1",
        block_size=2048,
        batch_size=1,
        streaming=True,
    )
    
    # Train (single GPU with AMP fp16)
    print("\nStarting training...")
    history = train_single_gpu(
        model=model,
        train_loader=train_loader,
        steps=1000,
        lr=3e-4,
        log_every=10,
        device=device,
    )
    
    print("\nTraining complete!")
    
    # Test generation
    print("\nTesting generation...")
    model.eval()
    
    prompt_text = "Once upon a time"
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    
    # Pad to block size
    if prompt_ids.size(1) < cfg.block_size:
        prompt_ids = torch.nn.functional.pad(
            prompt_ids, 
            (0, cfg.block_size - prompt_ids.size(1)),
            value=tokenizer.pad_token_id
        )
    
    with torch.no_grad():
        generated = generate_photon(
            model=model,
            input_ids=prompt_ids,
            max_new_tokens=cfg.block_size,
            temperature=0.8,
            top_k=50,
            use_latent_ar=True,
        )
    
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"\nGenerated: {generated_text[:500]}...")


if __name__ == "__main__":
    main()
