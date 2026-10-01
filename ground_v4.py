"""Targeted factual grounding for OrcaAI v4.

This stage starts from the latest pretrained v4 checkpoint, trains on a small
set of basic factual associations, and mixes in base-text replay to reduce
catastrophic forgetting. It saves a separate checkpoint.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model_v4 import OrcaAIv4
from tokenizer import LocalTokenizer


DATA_PATH = Path("data/grounding_v4.jsonl")
FINEWEB_PATH = Path("data/fineweb_v4.bin")
WIKI_PATH = Path("data/wikipedia_v4.bin")

LATEST_BASE_PATH = Path("checkpoints/model_v4.pt")
BEST_BASE_PATH = Path("checkpoints/model_v4_best.pt")
TOKENIZER_PATH = Path("checkpoints/tokenizer_v4.json")
OUTPUT_PATH = Path("checkpoints/model_v4_grounded.pt")

MAX_STEPS = 300
BATCH_SIZE = 8
LEARNING_RATE = 3e-6
VALIDATION_FRACTION = 0.15
PRINT_EVERY = 10
EVAL_EVERY = 20
PATIENCE = 5
MIN_IMPROVEMENT = 0.002

BASE_REPLAY_EVERY = 3
BASE_REPLAY_WEIGHT = 0.20

SEED = 2404


def load_rows():
    rows = []
    with DATA_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if len(rows) < 50:
        raise ValueError("Grounding dataset is unexpectedly small.")

    random.Random(SEED).shuffle(rows)
    split = max(1, int(len(rows) * VALIDATION_FRACTION))
    return rows[split:], rows[:split]


def encode_example(row, tokenizer, context_length):
    prompt_ids = tokenizer.encode(row["prompt"])
    response_ids = tokenizer.encode(
        row["response"],
        add_eos=True,
    )

    token_ids = prompt_ids + response_ids
    mask = [False] * len(prompt_ids) + [True] * len(response_ids)

    if len(token_ids) > context_length + 1:
        token_ids = token_ids[-(context_length + 1):]
        mask = mask[-(context_length + 1):]

    x = token_ids[:-1]
    next_ids = token_ids[1:]
    next_mask = mask[1:]

    labels = [
        token if enabled else -100
        for token, enabled in zip(next_ids, next_mask)
    ]

    return (
        torch.tensor(x, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def build_examples(rows, tokenizer, context_length):
    return [
        encode_example(row, tokenizer, context_length)
        for row in rows
    ]


def make_batch(
    examples,
    tokenizer,
    context_length,
    device,
):
    chosen = random.choices(
        examples,
        k=BATCH_SIZE,
    )

    x = torch.full(
        (BATCH_SIZE, context_length),
        tokenizer.pad_id,
        dtype=torch.long,
    )
    labels = torch.full(
        (BATCH_SIZE, context_length),
        -100,
        dtype=torch.long,
    )

    for row, (ids, row_labels) in enumerate(chosen):
        length = min(len(ids), context_length)
        x[row, :length] = ids[:length]
        labels[row, :length] = row_labels[:length]

    return x.to(device), labels.to(device)


def supervised_loss(model, x, labels):
    logits, _ = model(x)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        label_smoothing=0.01,
    )


def load_base_tokens(path):
    if not path.exists():
        return None
    return np.memmap(
        path,
        dtype=np.uint16,
        mode="r",
    )


def make_replay_batch(
    data,
    context_length,
    device,
):
    max_start = len(data) - context_length - 1
    starts = torch.randint(
        0,
        max_start + 1,
        (BATCH_SIZE,),
    )

    x = torch.stack([
        torch.from_numpy(
            np.array(
                data[i:i + context_length],
                dtype=np.int64,
            )
        )
        for i in starts
    ])
    y = torch.stack([
        torch.from_numpy(
            np.array(
                data[i + 1:i + context_length + 1],
                dtype=np.int64,
            )
        )
        for i in starts
    ])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(
    model,
    examples,
    tokenizer,
    context_length,
    device,
    use_bf16,
):
    model.eval()
    losses = []

    for _ in range(20):
        x, labels = make_batch(
            examples,
            tokenizer,
            context_length,
            device,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            loss = supervised_loss(
                model,
                x,
                labels,
            )
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def save_model(
    model,
    config,
    step,
    val_loss,
    source_step,
):
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "stage": "ground_v4",
            "step": step,
            "source_step": source_step,
            "val_loss": val_loss,
        },
        OUTPUT_PATH,
    )


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Missing data/grounding_v4.jsonl. "
            "Run: py -3.12 generate_grounding_v4.py"
        )

    base_path = (
        LATEST_BASE_PATH
        if LATEST_BASE_PATH.exists()
        else BEST_BASE_PATH
    )

    if not base_path.exists():
        raise FileNotFoundError(
            "No v4 base checkpoint found."
        )

    tokenizer = LocalTokenizer.load(
        TOKENIZER_PATH
    )

    checkpoint = torch.load(
        base_path,
        map_location="cpu",
        weights_only=False,
    )

    config = checkpoint["config"]
    context_length = config["context_length"]

    train_rows, val_rows = load_rows()
    train_examples = build_examples(
        train_rows,
        tokenizer,
        context_length,
    )
    val_examples = build_examples(
        val_rows,
        tokenizer,
        context_length,
    )

    fineweb = load_base_tokens(FINEWEB_PATH)
    wikipedia = load_base_tokens(WIKI_PATH)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_bf16 = (
        device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    model = OrcaAIv4(**config).to(device)
    model.load_state_dict(
        checkpoint["model_state"]
    )
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    print("=" * 72)
    print("OrcaAI v4 factual grounding")
    print("=" * 72)
    print(f"Base:         {base_path.name}")
    print(f"Base step:    {checkpoint.get('step', 'unknown')}")
    print(f"Train rows:   {len(train_rows)}")
    print(f"Val rows:     {len(val_rows)}")
    print(f"Steps:        {MAX_STEPS}")
    print(f"Learning rate:{LEARNING_RATE:.1e}")
    print()
    print("This saves a SEPARATE checkpoint and does not overwrite pretraining.")
    print()

    best_val = float("inf")
    stale = 0

    for step in range(1, MAX_STEPS + 1):
        x, labels = make_batch(
            train_examples,
            tokenizer,
            context_length,
            device,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            loss = supervised_loss(
                model,
                x,
                labels,
            )
            total_loss = loss

            if step % BASE_REPLAY_EVERY == 0:
                source = (
                    fineweb
                    if random.random() < 0.7
                    else wikipedia
                )

                if source is not None:
                    replay_x, replay_y = make_replay_batch(
                        source,
                        context_length,
                        device,
                    )
                    _, replay_loss = model(
                        replay_x,
                        replay_y,
                    )
                    total_loss = (
                        loss
                        + BASE_REPLAY_WEIGHT
                        * replay_loss
                    )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )
        optimizer.step()

        if step == 1 or step % PRINT_EVERY == 0:
            print(
                f"Step {step:>3}/{MAX_STEPS} | "
                f"ground {loss.item():.4f}"
            )

        if step % EVAL_EVERY == 0:
            val_loss = evaluate(
                model,
                val_examples,
                tokenizer,
                context_length,
                device,
                use_bf16,
            )
            print(
                f"  grounding val loss: {val_loss:.4f}"
            )

            if val_loss < best_val - MIN_IMPROVEMENT:
                best_val = val_loss
                stale = 0
                save_model(
                    model,
                    config,
                    step,
                    val_loss,
                    checkpoint.get("step"),
                )
                print(
                    "  new best grounded checkpoint saved"
                )
            else:
                stale += 1
                print(
                    f"  no meaningful improvement "
                    f"({stale}/{PATIENCE})"
                )

                if stale >= PATIENCE:
                    print("  early stopping")
                    break

    print()
    print("=" * 72)
    print("Grounding complete")
    print("=" * 72)
    print(f"Saved: {OUTPUT_PATH}")
    print()
    print(
        "Next: py -3.12 diagnose_v4.py --grounded --extended"
    )


if __name__ == "__main__":
    main()
