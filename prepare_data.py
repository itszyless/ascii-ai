"""Build LocalAI's pretraining dataset and train its tokenizer from scratch.

Default: WikiText-2 for a manageable first real pretraining run.
Use --full for WikiText-103 once the pipeline works.
"""

import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset

from tokenizer import LocalTokenizer


DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints")
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer.json"
TRAIN_BIN = DATA_DIR / "train.bin"
VAL_BIN = DATA_DIR / "val.bin"


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
        "--full",
        action="store_true",
        help="Use WikiText-103 instead of the much smaller WikiText-2.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=4096,
        help="Size of the BPE vocabulary trained from scratch.",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    config = "wikitext-103-raw-v1" if args.full else "wikitext-2-raw-v1"

    print("=" * 62)
    print("LocalAI data preparation")
    print("=" * 62)
    print(f"Dataset: Salesforce/wikitext ({config})")
    print("Downloading text only. No pretrained model weights are used.")
    print()

    dataset = load_dataset("Salesforce/wikitext", config)

    train_texts = list(clean_lines(dataset["train"]["text"]))
    val_texts = list(clean_lines(dataset["validation"]["text"]))

    print(f"Training documents/lines: {len(train_texts):,}")
    print("Training a byte-level BPE tokenizer from this corpus...")

    tokenizer = LocalTokenizer.train(
        train_texts,
        vocab_size=args.vocab_size,
    )
    tokenizer.save(TOKENIZER_PATH)

    print(f"Tokenizer vocabulary: {tokenizer.vocab_size:,}")
    print("Encoding dataset...")

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
