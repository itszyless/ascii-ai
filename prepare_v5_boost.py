"""Prepare OrcaAI v5 multilingual + math continuation data.

This stage deliberately reuses the existing v5 tokenizer so the currently
running 120k-step pretraining job does not need to restart.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import random

import numpy as np
from datasets import load_dataset

from tokenizer import LocalTokenizer


DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints")
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer_v5.json"

MULTI_TRAIN = DATA_DIR / "multilingual_v5_boost.bin"
MULTI_VAL = DATA_DIR / "multilingual_v5_boost_val.bin"
MATH_TRAIN = DATA_DIR / "math_v5_boost.bin"
MATH_VAL = DATA_DIR / "math_v5_boost_val.bin"

WIKI_NAME = "wikimedia/wikipedia"
WIKI_VERSION = "20231101"
SEED = 2505
SHUFFLE_BUFFER = 10_000

# German gets extra weight because it is a primary target language for OrcaAI.
LANGUAGE_TARGETS = {
    "de": 24_000_000,
    "fr": 12_000_000,
    "es": 12_000_000,
    "it": 12_000_000,
    "pt": 12_000_000,
    "nl": 10_000_000,
    "pl": 10_000_000,
    "cs": 8_000_000,
    "sv": 8_000_000,
}
LANGUAGE_VAL_TARGETS = {
    language: max(300_000, target // 24)
    for language, target in LANGUAGE_TARGETS.items()
}


def row_text(row):
    return (row.get("text") or "").strip()


def row_key(row, language):
    return f"{language}:{row.get('id') or row.get('url') or row.get('title') or row_text(row)[:300]}"


def is_validation(row, language):
    digest = hashlib.sha1(
        row_key(row, language).encode("utf-8", errors="ignore")
    ).digest()
    return int.from_bytes(digest[:4], "big") % 1000 < 25


def wikipedia_stream(language, seed_offset):
    dataset = load_dataset(
        WIKI_NAME,
        f"{WIKI_VERSION}.{language}",
        split="train",
        streaming=True,
    )
    return dataset.shuffle(
        seed=SEED + seed_offset,
        buffer_size=SHUFFLE_BUFFER,
    )


def append_language(tokenizer, language, train_target, val_target, offset):
    train_written = 0
    val_written = 0
    docs = 0

    with MULTI_TRAIN.open("ab") as train_file, MULTI_VAL.open("ab") as val_file:
        for row in wikipedia_stream(language, offset):
            text = row_text(row)
            if not text:
                continue

            ids = tokenizer.encode(text, add_eos=True)
            if not ids:
                continue

            if is_validation(row, language) and val_written < val_target:
                remaining = val_target - val_written
                array = np.asarray(ids[:remaining], dtype=np.uint16)
                val_file.write(array.tobytes())
                val_written += len(array)
            elif train_written < train_target:
                remaining = train_target - train_written
                array = np.asarray(ids[:remaining], dtype=np.uint16)
                train_file.write(array.tobytes())
                train_written += len(array)

            docs += 1
            if docs % 1500 == 0:
                print(
                    f"{language}: train {train_written:,}/{train_target:,} | "
                    f"val {val_written:,}/{val_target:,}"
                )

            if train_written >= train_target and val_written >= val_target:
                break

    if train_written < train_target or val_written < val_target:
        raise RuntimeError(
            f"{language} stream ended early: "
            f"{train_written:,}/{train_target:,} train, "
            f"{val_written:,}/{val_target:,} val"
        )

    return train_written, val_written


MATH_TEMPLATES = [
    ("Question: What is {a} + {b}?\nAnswer: {answer}.", "+"),
    ("Question: What is {a} - {b}?\nAnswer: {answer}.", "-"),
    ("Question: What is {a} × {b}?\nAnswer: {answer}.", "*"),
    ("Question: What is {a} times {b}?\nAnswer: {answer}.", "*"),
    ("Aufgabe: Wie viel ist {a} + {b}?\nAntwort: {answer}.", "+"),
    ("Aufgabe: Wie viel ist {a} - {b}?\nAntwort: {answer}.", "-"),
    ("Aufgabe: Wie viel ist {a} mal {b}?\nAntwort: {answer}.", "*"),
    ("Pregunta: ¿Cuánto es {a} × {b}?\nRespuesta: {answer}.", "*"),
    ("Question : combien font {a} × {b} ?\nRéponse : {answer}.", "*"),
    ("Domanda: quanto fa {a} × {b}?\nRisposta: {answer}.", "*"),
]


def make_math_example(rng):
    template, operation = rng.choice(MATH_TEMPLATES)

    if operation == "*":
        # Bias toward school-style multiplication while still showing larger
        # products than the tiny 0..12 table used by v4.
        a = rng.randint(0, 999)
        b = rng.randint(0, 99)
        answer = a * b
    elif operation == "+":
        a = rng.randint(0, 9999)
        b = rng.randint(0, 9999)
        answer = a + b
    else:
        a = rng.randint(0, 9999)
        b = rng.randint(0, a)
        answer = a - b

    return template.format(a=a, b=b, answer=answer)


def write_math(tokenizer, train_target, val_target):
    rng_train = random.Random(SEED + 100)
    rng_val = random.Random(SEED + 101)

    for path in (MATH_TRAIN, MATH_VAL):
        path.write_bytes(b"")

    def fill(path, target, rng, label):
        written = 0
        examples = 0
        with path.open("ab") as output:
            while written < target:
                text = make_math_example(rng)
                ids = tokenizer.encode(text, add_eos=True)
                remaining = target - written
                array = np.asarray(ids[:remaining], dtype=np.uint16)
                output.write(array.tobytes())
                written += len(array)
                examples += 1
                if examples % 25_000 == 0:
                    print(f"{label}: {written:,}/{target:,} tokens")
        return written, examples

    train_written, train_examples = fill(
        MATH_TRAIN, train_target, rng_train, "math train"
    )
    val_written, val_examples = fill(
        MATH_VAL, val_target, rng_val, "math val"
    )
    return train_written, val_written, train_examples, val_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--math-tokens", type=int, default=24_000_000)
    parser.add_argument("--math-val-tokens", type=int, default=2_000_000)
    args = parser.parse_args()

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            "Missing checkpoints/tokenizer_v5.json. "
            "Finish v5 preparation first."
        )

    DATA_DIR.mkdir(exist_ok=True)
    tokenizer = LocalTokenizer.load(TOKENIZER_PATH)

    MULTI_TRAIN.write_bytes(b"")
    MULTI_VAL.write_bytes(b"")

    print("=" * 72)
    print("OrcaAI v5 multilingual + math boost preparation")
    print("=" * 72)
    print("Reusing the existing v5 tokenizer — no base-training restart needed.")
    print()

    total_train = 0
    total_val = 0
    for offset, (language, target) in enumerate(LANGUAGE_TARGETS.items()):
        val_target = LANGUAGE_VAL_TARGETS[language]
        print(
            f"Encoding {language}: {target:,} train + "
            f"{val_target:,} val tokens"
        )
        train_written, val_written = append_language(
            tokenizer,
            language,
            target,
            val_target,
            offset,
        )
        total_train += train_written
        total_val += val_written

    print()
    math_train, math_val, math_examples, math_val_examples = write_math(
        tokenizer,
        args.math_tokens,
        args.math_val_tokens,
    )

    print()
    print("=" * 72)
    print("Boost data ready")
    print("=" * 72)
    print(f"Multilingual train: {total_train:,} tokens")
    print(f"Multilingual val:   {total_val:,} tokens")
    print(
        f"Math train:         {math_train:,} tokens "
        f"({math_examples:,} generated examples)"
    )
    print(
        f"Math val:           {math_val:,} tokens "
        f"({math_val_examples:,} generated examples)"
    )
    print()
    print("After base v5 reaches 120k:")
    print("  py -3.12 orca.py boost-v5")


if __name__ == "__main__":
    main()
