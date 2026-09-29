"""Prepare the mixed OrcaAI v4 pretraining corpus.

v4 combines high-quality educational web text with cleaned English Wikipedia.
Train/validation membership is deterministic by document identity so the two
splits are disjoint within each source.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from datasets import load_dataset

from tokenizer import LocalTokenizer


DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints")
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer_v4.json"

FINEWEB_TRAIN = DATA_DIR / "fineweb_v4.bin"
FINEWEB_VAL = DATA_DIR / "fineweb_v4_val.bin"
WIKI_TRAIN = DATA_DIR / "wikipedia_v4.bin"
WIKI_VAL = DATA_DIR / "wikipedia_v4_val.bin"

FINEWEB_NAME = "HuggingFaceFW/fineweb-edu"
FINEWEB_CONFIG = "sample-10BT"
WIKI_NAME = "wikimedia/wikipedia"
WIKI_CONFIG = "20231101.en"

SEED = 2404
SHUFFLE_BUFFER = 20_000


def fineweb_stream():
    dataset = load_dataset(
        FINEWEB_NAME,
        FINEWEB_CONFIG,
        split="train",
        streaming=True,
    )
    return dataset.shuffle(
        seed=SEED,
        buffer_size=SHUFFLE_BUFFER,
    )


def wikipedia_stream():
    dataset = load_dataset(
        WIKI_NAME,
        WIKI_CONFIG,
        split="train",
        streaming=True,
    )
    return dataset.shuffle(
        seed=SEED + 1,
        buffer_size=SHUFFLE_BUFFER,
    )


def row_text(row):
    return (row.get("text") or "").strip()


def row_key(row):
    return str(
        row.get("id")
        or row.get("url")
        or row.get("title")
        or row_text(row)[:300]
    )


def is_validation(row, per_thousand=20):
    digest = hashlib.sha1(
        row_key(row).encode("utf-8", errors="ignore")
    ).digest()
    bucket = int.from_bytes(digest[:4], "big") % 1000
    return bucket < per_thousand


def tokenizer_texts(fineweb_docs, wiki_docs):
    fine = iter(fineweb_stream())
    wiki = iter(wikipedia_stream())

    fine_count = 0
    wiki_count = 0

    while fine_count < fineweb_docs or wiki_count < wiki_docs:
        if fine_count < fineweb_docs:
            try:
                text = row_text(next(fine))
                if text:
                    fine_count += 1
                    yield text
            except StopIteration:
                fine_count = fineweb_docs

        if wiki_count < wiki_docs:
            try:
                text = row_text(next(wiki))
                if text:
                    wiki_count += 1
                    yield text
            except StopIteration:
                wiki_count = wiki_docs


def write_split_source(
    iterator,
    tokenizer,
    train_path,
    val_path,
    train_target,
    val_target,
    label,
):
    train_written = 0
    val_written = 0
    documents = 0

    with train_path.open("wb") as train_file, val_path.open("wb") as val_file:
        for row in iterator:
            text = row_text(row)
            if not text:
                continue

            ids = tokenizer.encode(text, add_eos=True)
            if not ids:
                continue

            target_val = is_validation(row)
            if target_val and val_written < val_target:
                remaining = val_target - val_written
                array = np.asarray(ids[:remaining], dtype=np.uint16)
                val_file.write(array.tobytes())
                val_written += len(array)
            elif (not target_val) and train_written < train_target:
                remaining = train_target - train_written
                array = np.asarray(ids[:remaining], dtype=np.uint16)
                train_file.write(array.tobytes())
                train_written += len(array)

            documents += 1

            if documents % 2000 == 0:
                print(
                    f"{label}: train {train_written:,}/{train_target:,} | "
                    f"val {val_written:,}/{val_target:,} | "
                    f"{documents:,} docs"
                )

            if (
                train_written >= train_target
                and val_written >= val_target
            ):
                break

    if train_written < train_target or val_written < val_target:
        raise RuntimeError(
            f"{label} stream ended early. "
            f"train={train_written:,}/{train_target:,}, "
            f"val={val_written:,}/{val_target:,}"
        )

    return train_written, val_written, documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fineweb-tokens",
        type=int,
        default=350_000_000,
    )
    parser.add_argument(
        "--wiki-tokens",
        type=int,
        default=150_000_000,
    )
    parser.add_argument(
        "--fineweb-val-tokens",
        type=int,
        default=7_000_000,
    )
    parser.add_argument(
        "--wiki-val-tokens",
        type=int,
        default=3_000_000,
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=12_000,
    )
    parser.add_argument(
        "--fineweb-tokenizer-docs",
        type=int,
        default=40_000,
    )
    parser.add_argument(
        "--wiki-tokenizer-docs",
        type=int,
        default=20_000,
    )
    args = parser.parse_args()

    if args.vocab_size > 65535:
        raise ValueError(
            "vocab-size must stay <= 65535 because token files use uint16."
        )

    DATA_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    print("=" * 72)
    print("OrcaAI v4 data preparation")
    print("=" * 72)
    print(
        f"FineWeb-Edu: {args.fineweb_tokens:,} train + "
        f"{args.fineweb_val_tokens:,} val tokens"
    )
    print(
        f"Wikipedia:   {args.wiki_tokens:,} train + "
        f"{args.wiki_val_tokens:,} val tokens"
    )
    print(f"Tokenizer:   {args.vocab_size:,} tokens")
    print()
    print("Training a fresh v4 tokenizer...")

    tokenizer = LocalTokenizer.train(
        tokenizer_texts(
            args.fineweb_tokenizer_docs,
            args.wiki_tokenizer_docs,
        ),
        vocab_size=args.vocab_size,
    )
    tokenizer.save(TOKENIZER_PATH)

    print(
        f"Tokenizer saved: {TOKENIZER_PATH} "
        f"({tokenizer.vocab_size:,} tokens)"
    )
    print()

    print("Encoding FineWeb-Edu...")
    fw_train, fw_val, fw_docs = write_split_source(
        fineweb_stream(),
        tokenizer,
        FINEWEB_TRAIN,
        FINEWEB_VAL,
        args.fineweb_tokens,
        args.fineweb_val_tokens,
        "FineWeb",
    )

    print()
    print("Encoding Wikipedia...")
    wiki_train, wiki_val, wiki_docs = write_split_source(
        wikipedia_stream(),
        tokenizer,
        WIKI_TRAIN,
        WIKI_VAL,
        args.wiki_tokens,
        args.wiki_val_tokens,
        "Wikipedia",
    )

    print()
    print("=" * 72)
    print("OrcaAI v4 preparation complete")
    print("=" * 72)
    print(
        f"FineWeb:   {fw_train:,} train / {fw_val:,} val "
        f"from {fw_docs:,} documents"
    )
    print(
        f"Wikipedia: {wiki_train:,} train / {wiki_val:,} val "
        f"from {wiki_docs:,} documents"
    )
    print(
        f"Total train tokens: {fw_train + wiki_train:,}"
    )
    print(
        f"Total val tokens:   {fw_val + wiki_val:,}"
    )
    print()
    print("Next: py -3.12 train_v4.py")


if __name__ == "__main__":
    main()
