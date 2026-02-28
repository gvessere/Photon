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
    eval_from_train_examples: int = 10000,
    train_skip_examples: int = 0,
) -> tuple:
    """
    Create train and eval dataloaders.
    
    Args:
        dataset_name: HuggingFace dataset name
        tokenizer_name: HuggingFace tokenizer name
        block_size: Sequence length (should be divisible by C1*C2)
        batch_size: Batch size
        num_workers: DataLoader workers
        streaming: Use streaming dataset
        train_split: Training split name
        eval_split: Eval split name. If missing in dataset, derive eval from train split.
        eval_from_train_examples: Number of examples to reserve for eval when
            eval_split is unavailable and streaming=True.
        train_skip_examples: Number of token-block training examples to skip
            from the start of the processed train stream (used for resume).
    
    Returns:
        (train_loader, eval_loader, tokenizer)
    """
    from datasets import load_dataset
    
    # Create tokenizer
    tokenizer = create_tokenizer(tokenizer_name)
    eos_token_id = tokenizer.eos_token_id
    
    # Load training split and ensure eval data exists.
    train_dataset = load_dataset(dataset_name, split=train_split, streaming=streaming)
    eval_dataset = None
    if eval_split:
        try:
            eval_dataset = load_dataset(dataset_name, split=eval_split, streaming=streaming)
        except ValueError as e:
            if streaming:
                if eval_from_train_examples <= 0:
                    raise ValueError(
                        "eval_from_train_examples must be > 0 when deriving eval from train split."
                    ) from e
                # Deterministic split for iterable datasets: reserve the first N examples for eval.
                eval_dataset = load_dataset(dataset_name, split=train_split, streaming=True).take(
                    eval_from_train_examples
                )
                train_dataset = train_dataset.skip(eval_from_train_examples)
            else:
                split_ds = train_dataset.train_test_split(test_size=0.01, seed=42, shuffle=True)
                train_dataset = split_ds["train"]
                eval_dataset = split_ds["test"]
    
    # Tokenize
    tokenize_partial = partial(tokenize_fn, tokenizer=tokenizer, max_length=block_size)
    tokenized = train_dataset.map(
        tokenize_partial,
        batched=True,
        remove_columns=["text", "meta"] if "meta" in train_dataset.column_names else ["text"]
    )
    
    # Group into blocks
    group_partial = partial(group_texts, block_size=block_size, eos_token_id=eos_token_id)
    lm_dataset = tokenized.map(group_partial, batched=True)

    # Fast-forward processed train stream for checkpoint resume.
    if train_skip_examples > 0:
        if hasattr(lm_dataset, "skip"):
            lm_dataset = lm_dataset.skip(train_skip_examples)
        else:
            raise ValueError("train_skip_examples requires a dataset supporting .skip()")
    
    # Create dataloader
    train_loader = DataLoader(
        lm_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    if eval_dataset is None:
        raise ValueError(
            "Evaluation dataset could not be created. Set --eval_split or provide a dataset with a valid eval split."
        )

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
