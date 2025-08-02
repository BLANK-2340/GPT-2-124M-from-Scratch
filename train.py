#!/usr/bin/env python3
"""
Train a GPT‑2 language model from scratch.

This script builds a GPT‑2 model configuration according to user‑specified
hyper‑parameters (number of layers, attention heads, embedding size, etc.) and
fine‑tunes it on a text dataset.  You can choose between loading a dataset
from the Hugging Face `datasets` library or providing a local plain‑text file.

Example usage:

    # Train on WikiText‑2 with default hyper‑parameters
    python train.py

    # Train on a custom corpus with 6 layers and larger context
    python train.py --train_file path/to/corpus.txt --n_layers 6 --block_size 256 --epochs 3

The script uses Hugging Face's `Trainer` for simplicity.  While it is not
optimised for speed, it is easy to modify for experimentation.
"""

import argparse
import logging
import os
from typing import Optional

import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

try:
    from datasets import load_dataset, Dataset
except ImportError as e:
    raise SystemExit(
        "The `datasets` library is required for this script. Please install it via `pip install datasets`."
    ) from e


logger = logging.getLogger(__name__)


def get_arguments() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(description="Train a GPT‑2 model from scratch.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="Path to a plain‑text file for training.  If provided, this file will be used instead of a Hugging Face dataset.",
    )
    group.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="Name of a dataset on the Hugging Face hub (default: 'wikitext').",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Configuration name of the dataset (ignored when using a train_file).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./model_output",
        help="Directory where the model checkpoints and final weights will be saved.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs (default: 1).")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Per‑device batch size for training.  Reduce if you encounter CUDA OOM errors.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=128,
        help="Maximum sequence length (context size).  GPT‑2 uses positional embeddings up to this length.",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=12,
        help="Number of transformer layers in the model (default: 12).",
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=12,
        help="Number of attention heads per layer (default: 12).",
    )
    parser.add_argument(
        "--n_embd",
        type=int,
        default=768,
        help="Dimensionality of the embeddings and hidden states (default: 768).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Learning rate for the AdamW optimizer (default: 5e-4).",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Interval (in steps) at which to log training progress.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Interval (in steps) at which to save model checkpoints.",
    )
    return parser.parse_args()


def prepare_dataset(
    tokenizer: GPT2TokenizerFast,
    block_size: int,
    train_file: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_config: Optional[str] = None,
) -> Dataset:
    """
    Load and tokenize the training dataset.

    If `train_file` is provided, read it line by line and tokenize.  Otherwise,
    download the specified dataset from the Hugging Face hub.

    Args:
        tokenizer: The tokenizer used to convert text to token IDs.
        block_size: The maximum length of a token sequence (model context).
        train_file: Optional path to a local training corpus.
        dataset_name: Optional name of a Hugging Face dataset.
        dataset_config: Optional configuration for the dataset.

    Returns:
        A `datasets.Dataset` ready for use with the Trainer.
    """
    if train_file:
        if not os.path.isfile(train_file):
            raise FileNotFoundError(f"Training file '{train_file}' not found.")

        # Create a dataset from a text file.  Each line becomes a separate example.
        logger.info("Loading dataset from local file %s", train_file)
        with open(train_file, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f]
        dataset = Dataset.from_dict({"text": lines})
    else:
        logger.info("Loading dataset %s (%s) from Hugging Face hub", dataset_name, dataset_config)
        dataset = load_dataset(dataset_name, dataset_config, split="train")

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=block_size,
            return_special_tokens_mask=True,
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=os.cpu_count() or 1, remove_columns=["text"])
    return tokenized_dataset


def main() -> None:
    args = get_arguments()

    # Set up basic logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.info("Training arguments: %s", args)

    # Prepare tokenizer; we reuse the GPT‑2 tokenizer for convenience.  It includes
    # special tokens and a BPE vocabulary.
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # Add padding token if not present (GPT‑2 was trained without padding token)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # Prepare dataset
    tokenized_dataset = prepare_dataset(
        tokenizer,
        block_size=args.block_size,
        train_file=args.train_file,
        dataset_name=None if args.train_file else args.dataset_name,
        dataset_config=args.dataset_config,
    )

    # Build GPT‑2 configuration and model from scratch
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=args.block_size,
        n_ctx=args.block_size,
        n_embd=args.n_embd,
        n_layer=args.n_layers,
        n_head=args.n_heads,
    )
    model = GPT2LMHeadModel(config)
    # Resize token embeddings in case we added a pad token
    model.resize_token_embeddings(len(tokenizer))

    # Use DataCollator to handle dynamic padding and creation of labels
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT‑2 is a causal LM
    )

    # TrainingArguments define hyper‑parameters for the Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        warmup_steps=0,
        prediction_loss_only=True,
        report_to=[],  # Disable logging to third‑party services
    )

    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train!
    logger.info("Starting training")
    trainer.train()

    # Save final model and tokenizer
    logger.info("Saving model to %s", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()