"""
PHOTON Data Pipeline

Handles:
- Tokenization with hierarchical padding
- Block-based grouping for chunk-aligned sequences
- EOS insertion between documents
- Collation for training
"""

from typing import Dict, List, Optional, Any, Iterator
from functools import partial

import torch
from torch.utils.data import DataLoader, IterableDataset


def create_tokenizer(model_name: str = "mistralai/Mistral-7B-v0.1"):
    """Create and configure tokenizer."""
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer


def tokenize_fn(example: Dict[str, Any], tokenizer, max_length: int = 2048) -> Dict[str, List[int]]:
    """Tokenize a single example."""
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        return_attention_mask=False
    )


def group_texts(
    examples: Dict[str, List],
    block_size: int,
    eos_token_id: int
) -> Dict[str, List]:
    """
    Group tokenized examples into fixed-size blocks.
    
    - Inserts EOS between documents
    - Concatenates into a stream
    - Splits into block_size chunks
    - Creates labels (same as input_ids for LM)
    
    Args:
        examples: Batched examples with "input_ids" key
        block_size: Size of each output block (should be divisible by C1*C2)
        eos_token_id: Token ID for EOS separator
    
    Returns:
        Dict with "input_ids" and "labels" lists
    """
    stream = {"input_ids": []}
    
    n = len(examples["input_ids"])
    for i in range(n):
        stream["input_ids"].extend(examples["input_ids"][i])
        stream["input_ids"].append(eos_token_id)
    
    # Truncate to multiple of block_size
    total_len = (len(stream["input_ids"]) // block_size) * block_size
    
    if total_len == 0:
        return {"input_ids": [], "labels": []}
    
    result = {
        "input_ids": [
            stream["input_ids"][i:i + block_size]
            for i in range(0, total_len, block_size)
        ]
    }
    result["labels"] = result["input_ids"].copy()
    
    return result


def collate_fn(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    """
    Collate batch of examples into tensors.
    
    Args:
        batch: List of dicts with "input_ids" and "labels"
    
    Returns:
        Dict with tensor "input_ids" and "labels"
    """
    input_ids = torch.tensor([x["input_ids"] for x in batch], dtype=torch.long)
    labels = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
    return {"input_ids": input_ids, "labels": labels}


def create_dataloaders(
    dataset_name: str = "EleutherAI/the_pile_deduplicated",
    tokenizer_name: str = "mistralai/Mistral-7B-v0.1",
    block_size: int = 2048,
    batch_size: int = 8,
    num_workers: int = 0,
    streaming: bool = True,
    train_split: str = "train",
    eval_split: Optional[str] = None,
) -> tuple:
    """
    Create train (and optionally eval) dataloaders.
    
    Args:
        dataset_name: HuggingFace dataset name
        tokenizer_name: HuggingFace tokenizer name
        block_size: Sequence length (should be divisible by C1*C2)
        batch_size: Batch size
        num_workers: DataLoader workers
        streaming: Use streaming dataset
        train_split: Training split name
        eval_split: Optional eval split name
    
    Returns:
        (train_loader, eval_loader, tokenizer) - eval_loader may be None
    """
    from datasets import load_dataset
    
    # Create tokenizer
    tokenizer = create_tokenizer(tokenizer_name)
    eos_token_id = tokenizer.eos_token_id
    
    # Load dataset
    dataset = load_dataset(dataset_name, split=train_split, streaming=streaming)
    
    # Tokenize
    tokenize_partial = partial(tokenize_fn, tokenizer=tokenizer, max_length=block_size)
    tokenized = dataset.map(
        tokenize_partial,
        batched=True,
        remove_columns=["text", "meta"] if "meta" in dataset.column_names else ["text"]
    )
    
    # Group into blocks
    group_partial = partial(group_texts, block_size=block_size, eos_token_id=eos_token_id)
    lm_dataset = tokenized.map(group_partial, batched=True)
    
    # Create dataloader
    train_loader = DataLoader(
        lm_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    # Eval loader if requested
    eval_loader = None
    if eval_split:
        eval_dataset = load_dataset(dataset_name, split=eval_split, streaming=streaming)
        eval_tokenized = eval_dataset.map(
            tokenize_partial,
            batched=True,
            remove_columns=["text", "meta"] if "meta" in eval_dataset.column_names else ["text"]
        )
        eval_lm = eval_tokenized.map(group_partial, batched=True)
        eval_loader = DataLoader(
            eval_lm,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers
        )
    
    return train_loader, eval_loader, tokenizer


class PhotonDataset(IterableDataset):
    """
    Custom iterable dataset for PHOTON training.
    
    Handles streaming from HuggingFace datasets with proper
    block alignment for hierarchical chunking.
    """
    
    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        block_size: int,
        split: str = "train",
        max_length: int = 2048,
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.split = split
        self.max_length = max_length
        self.eos_token_id = tokenizer.eos_token_id
    
    def __iter__(self) -> Iterator[Dict[str, List[int]]]:
        from datasets import load_dataset
        
        dataset = load_dataset(self.dataset_name, split=self.split, streaming=True)
        
        buffer = []
        for example in dataset:
            # Tokenize
            tokens = self.tokenizer(
                example["text"],
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=False
            )["input_ids"]
            
            buffer.extend(tokens)
            buffer.append(self.eos_token_id)
            
            # Yield complete blocks
            while len(buffer) >= self.block_size:
                block = buffer[:self.block_size]
                buffer = buffer[self.block_size:]
                yield {"input_ids": block, "labels": block.copy()}
