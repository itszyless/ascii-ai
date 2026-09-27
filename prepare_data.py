"""Prepare the v2 LocalAI pretraining corpus and tokenizer from scratch.

v2 intentionally uses separate files so the old experiment and chat.txt stay safe.
By default it uses WikiText-103 instead of the tiny WikiText-2 smoke-test corpus.
"""

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset

from tokenizer import LocalTokenizer


DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints")
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer_v2.json"
TRAIN_BIN = DATA_DIR / "train_v2.bin"
VAL_BIN = DATA_DIR / "val_v2.bin"


def clean_lines(lines):
    for line in lines:
        text = line.strip()
        if text:
            yield text + "\n"


def write_tokens(path, texts, tokenizer):
    total = 0
    with path.open("wb") as file:
        for text in texts:
            ids = tokenizer.encode(text, add_eos=True)
            array = np.asarray(ids, dtype=np.uint16)
            file.write(array.tobytes())
            total += len(ids)
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use WikiText-2 only for a quick smoke test instead of the v2 corpus.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8192,
        help="BPE vocabulary size. v2 default: 8192.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    config = "wikitext-2-raw-v1" if args.small else "wikitext-103-raw-v1"

    print("=" * 66)
    print("LocalAI v2 data preparation")
    print("=" * 66)
    print(f"Dataset: Salesforce/wikitext ({config})")
    print("Text only; no pretrained model weights are downloaded.")
    print(f"Tokenizer vocabulary target: {args.vocab_size:,}")
    print()

    dataset = load_dataset("Salesforce/wikitext", config)
    train_texts = list(clean_lines(dataset["train"]["text"]))
    val_texts = list(clean_lines(dataset["validation"]["text"]))

    print(f"Training documents/lines: {len(train_texts):,}")
    print("Training a fresh byte-level BPE tokenizer...")

    tokenizer = LocalTokenizer.train(train_texts, vocab_size=args.vocab_size)
    tokenizer.save(TOKENIZER_PATH)

    print(f"Actual tokenizer vocabulary: {tokenizer.vocab_size:,}")
    print("Encoding corpus...")

    train_tokens = write_tokens(TRAIN_BIN, train_texts, tokenizer)
    val_tokens = write_tokens(VAL_BIN, val_texts, tokenizer)

    print()
    print(f"Train tokens:      {train_tokens:,}")
    print(f"Validation tokens: {val_tokens:,}")
    print(f"Tokenizer:         {TOKENIZER_PATH}")
    print(f"Training data:     {TRAIN_BIN}")
    print(f"Validation data:   {VAL_BIN}")
    print()
    print("Ready. Run: py -3.12 train.py")


if __name__ == "__main__":
    main()
