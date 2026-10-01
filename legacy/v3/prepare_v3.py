"""Prepare OrcaAI v3 data from a much larger educational web corpus.

This keeps all v1/v2 files untouched.

Default target:
- 250M train tokens
- 5M validation tokens
- fresh 8192-token byte-level BPE tokenizer

Source: HuggingFaceFW/fineweb-edu, config sample-10BT, streamed so we do
not download the entire 10B-token sample.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset

from tokenizer import LocalTokenizer


DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints")
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer_v3.json"
TRAIN_BIN = DATA_DIR / "train_v3.bin"
VAL_BIN = DATA_DIR / "val_v3.bin"

DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
TOKENIZER_DOCUMENTS = 50_000
SHUFFLE_BUFFER = 10_000
SEED = 1337


def stream_dataset(seed):
    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train",
        streaming=True,
    )
    return dataset.shuffle(seed=seed, buffer_size=SHUFFLE_BUFFER)


def tokenizer_texts(limit):
    count = 0
    for row in stream_dataset(SEED):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        yield text
        count += 1
        if count >= limit:
            break


def write_token_stream(path, tokenizer, target_tokens, iterator, label):
    written = 0
    documents = 0

    with path.open("wb") as file:
        for row in iterator:
            text = (row.get("text") or "").strip()
            if not text:
                continue

            ids = tokenizer.encode(text, add_eos=True)
            if not ids:
                continue

            remaining = target_tokens - written
            if remaining <= 0:
                break

            if len(ids) > remaining:
                ids = ids[:remaining]

            array = np.asarray(ids, dtype=np.uint16)
            file.write(array.tobytes())
            written += len(ids)
            documents += 1

            if documents % 1000 == 0:
                pct = written / target_tokens * 100
                print(
                    f"{label}: {written:,}/{target_tokens:,} tokens "
                    f"({pct:.1f}%) | {documents:,} docs"
                )

            if written >= target_tokens:
                break

    if written < target_tokens:
        raise RuntimeError(
            f"Dataset stream ended after {written:,} {label} tokens; "
            f"target was {target_tokens:,}."
        )

    return written, documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-tokens",
        type=int,
        default=250_000_000,
        help="Number of pretraining tokens to write. Default: 250M.",
    )
    parser.add_argument(
        "--val-tokens",
        type=int,
        default=5_000_000,
        help="Number of validation tokens to write. Default: 5M.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8192,
        help="Fresh BPE vocabulary size. Default: 8192.",
    )
    parser.add_argument(
        "--tokenizer-docs",
        type=int,
        default=TOKENIZER_DOCUMENTS,
        help="Documents used to train the tokenizer.",
    )
    args = parser.parse_args()

    if args.vocab_size > 65535:
        raise ValueError("vocab-size must stay <= 65535 because token files use uint16.")

    DATA_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("OrcaAI v3 data preparation")
    print("=" * 70)
    print(f"Source:          {DATASET_NAME} ({DATASET_CONFIG})")
    print(f"Train target:    {args.train_tokens:,} tokens")
    print(f"Validation:      {args.val_tokens:,} tokens")
    print(f"Tokenizer docs:  {args.tokenizer_docs:,}")
    print(f"Vocab target:    {args.vocab_size:,}")
    print()
    print("Streaming text only. No pretrained model weights are downloaded.")
    print()

    print("Training fresh tokenizer...")
    tokenizer = LocalTokenizer.train(
        tokenizer_texts(args.tokenizer_docs),
        vocab_size=args.vocab_size,
    )
    tokenizer.save(TOKENIZER_PATH)
    print(f"Tokenizer saved: {TOKENIZER_PATH}")
    print(f"Actual vocabulary: {tokenizer.vocab_size:,}")
    print()

    # Separate shuffle seeds reduce train/validation overlap in practice.
    print("Writing validation tokens...")
    val_written, val_docs = write_token_stream(
        VAL_BIN,
        tokenizer,
        args.val_tokens,
        stream_dataset(SEED + 1),
        "Validation",
    )

    print()
    print("Writing training tokens...")
    train_written, train_docs = write_token_stream(
        TRAIN_BIN,
        tokenizer,
        args.train_tokens,
        stream_dataset(SEED + 2),
        "Train",
    )

    print()
    print("=" * 70)
    print("v3 preparation complete")
    print("=" * 70)
    print(f"Train:      {train_written:,} tokens from {train_docs:,} docs")
    print(f"Validation: {val_written:,} tokens from {val_docs:,} docs")
    print(f"Train file: {TRAIN_BIN} (~{TRAIN_BIN.stat().st_size / 1e6:.1f} MB)")
    print(f"Val file:   {VAL_BIN} (~{VAL_BIN.stat().st_size / 1e6:.1f} MB)")
    print(f"Tokenizer:  {TOKENIZER_PATH}")
    print()
    print("Next: py -3.12 train_v3.py")


if __name__ == "__main__":
    main()
