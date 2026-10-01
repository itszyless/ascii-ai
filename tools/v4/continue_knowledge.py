"""Long unattended knowledge-focused continued pretraining for OrcaAI v4.

Starts from the grounded v4 checkpoint when available, otherwise latest base.
Uses mostly Wikipedia with some FineWeb-Edu replay, a low learning rate, and
saves to a separate checkpoint so previous models stay untouched.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from model_v4 import OrcaAIv4
from tokenizer import LocalTokenizer


DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints")

FINEWEB_TRAIN = DATA_DIR / "fineweb_v4.bin"
FINEWEB_VAL = DATA_DIR / "fineweb_v4_val.bin"
WIKI_TRAIN = DATA_DIR / "wikipedia_v4.bin"
WIKI_VAL = DATA_DIR / "wikipedia_v4_val.bin"

TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer_v4.json"
GROUNDED_PATH = CHECKPOINT_DIR / "model_v4_grounded.pt"
LATEST_PATH = CHECKPOINT_DIR / "model_v4.pt"

OUTPUT_PATH = CHECKPOINT_DIR / "model_v4_knowledge.pt"
BEST_OUTPUT_PATH = CHECKPOINT_DIR / "model_v4_knowledge_best.pt"

STEPS = 20_000
MICRO_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2

WIKI_WEIGHT = 0.80
FINEWEB_WEIGHT = 0.20

LEARNING_RATE = 1.2e-5
MIN_LEARNING_RATE = 3e-6
WARMUP_STEPS = 300

PRINT_EVERY = 25
EVAL_EVERY = 500
EVAL_BATCHES = 12
SAVE_EVERY = 500

SEED = 2405


def load_tokens(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run prepare_v4.py first."
        )
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data, context_length, device):
    max_start = len(data) - context_length - 1
    if max_start < 0:
        raise ValueError("Dataset is too small.")

    starts = torch.randint(
        0,
        max_start + 1,
        (MICRO_BATCH_SIZE,),
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


def learning_rate(step):
    if step <= WARMUP_STEPS:
        return LEARNING_RATE * step / WARMUP_STEPS

    progress = (step - WARMUP_STEPS) / max(
        STEPS - WARMUP_STEPS,
        1,
    )
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    return MIN_LEARNING_RATE + cosine * (
        LEARNING_RATE - MIN_LEARNING_RATE
    )


@torch.no_grad()
def evaluate_source(
    model,
    data,
    context_length,
    device,
    use_bf16,
):
    model.eval()
    losses = []

    for _ in range(EVAL_BATCHES):
        x, y = get_batch(
            data,
            context_length,
            device,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            _, loss = model(x, y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(
    path,
    model,
    optimizer,
    step,
    config,
    best_val,
    source_name,
    source_step,
):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "stage": "knowledge_v4",
            "step": step,
            "source_name": source_name,
            "source_step": source_step,
            "best_val": best_val,
        },
        path,
    )


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    tokenizer = LocalTokenizer.load(TOKENIZER_PATH)

    fineweb_train = load_tokens(FINEWEB_TRAIN)
    fineweb_val = load_tokens(FINEWEB_VAL)
    wiki_train = load_tokens(WIKI_TRAIN)
    wiki_val = load_tokens(WIKI_VAL)

    source_path = (
        GROUNDED_PATH
        if GROUNDED_PATH.exists()
        else LATEST_PATH
    )
    if not source_path.exists():
        raise FileNotFoundError(
            "Need model_v4_grounded.pt or model_v4.pt."
        )

    source = torch.load(
        source_path,
        map_location="cpu",
        weights_only=False,
    )
    config = source["config"]
    context_length = config["context_length"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_bf16 = (
        device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    model = OrcaAIv4(**config).to(device)
    model.load_state_dict(source["model_state"])
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.05,
    )

    start_step = 0
    best_val = float("inf")

    if OUTPUT_PATH.exists():
        resume = torch.load(
            OUTPUT_PATH,
            map_location=device,
            weights_only=False,
        )
        if (
            resume.get("stage") == "knowledge_v4"
            and resume.get("config") == config
        ):
            model.load_state_dict(resume["model_state"])
            optimizer.load_state_dict(resume["optimizer_state"])
            start_step = resume.get("step", 0)
            best_val = resume.get("best_val", float("inf"))
            print(
                f"Resuming knowledge run from step {start_step:,}"
            )

    tokens_per_step = (
        MICRO_BATCH_SIZE
        * context_length
        * GRAD_ACCUM_STEPS
    )

    print("=" * 72)
    print("OrcaAI v4 knowledge-focused continued pretraining")
    print("=" * 72)
    print(f"Source checkpoint: {source_path.name}")
    print(
        f"Source step:       "
        f"{source.get('source_step', source.get('step', 'unknown'))}"
    )
    print(f"Device:            {device}")
    if device.type == "cuda":
        print(
            f"GPU:               "
            f"{torch.cuda.get_device_name(0)}"
        )
    print(f"Steps:             {STEPS:,}")
    print(
        f"Data mix:          "
        f"{WIKI_WEIGHT:.0%} Wikipedia / "
        f"{FINEWEB_WEIGHT:.0%} FineWeb-Edu"
    )
    print(
        f"Token positions:   "
        f"{STEPS * tokens_per_step:,}"
    )
    print()
    print(
        "This writes separate knowledge checkpoints. "
        "Your base and grounded checkpoints stay untouched."
    )
    print()

    started = time.time()

    for step in range(start_step + 1, STEPS + 1):
        lr = learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0

        for _ in range(GRAD_ACCUM_STEPS):
            data = (
                wiki_train
                if random.random() < WIKI_WEIGHT
                else fineweb_train
            )

            x, y = get_batch(
                data,
                context_length,
                device,
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                _, loss = model(x, y)
                scaled_loss = (
                    loss / GRAD_ACCUM_STEPS
                )

            scaled_loss.backward()
            train_loss += (
                loss.item() / GRAD_ACCUM_STEPS
            )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )
        optimizer.step()

        if step == start_step + 1 or step % PRINT_EVERY == 0:
            elapsed = max(
                time.time() - started,
                1e-6,
            )
            completed = step - start_step
            steps_per_second = completed / elapsed
            tok_per_second = (
                steps_per_second * tokens_per_step
            )
            remaining = STEPS - step
            eta_hours = (
                remaining
                / max(steps_per_second, 1e-9)
                / 3600
            )

            print(
                f"Step {step:>5}/{STEPS} | "
                f"train {train_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"{tok_per_second:,.0f} tok/s | "
                f"ETA {eta_hours:.2f}h"
            )

        if step % EVAL_EVERY == 0:
            wiki_loss = evaluate_source(
                model,
                wiki_val,
                context_length,
                device,
                use_bf16,
            )
            fineweb_loss = evaluate_source(
                model,
                fineweb_val,
                context_length,
                device,
                use_bf16,
            )

            combined = (
                WIKI_WEIGHT * wiki_loss
                + FINEWEB_WEIGHT * fineweb_loss
            )

            print(
                f"  validation {combined:.4f} | "
                f"Wikipedia {wiki_loss:.4f} | "
                f"FineWeb {fineweb_loss:.4f}"
            )

            if combined < best_val:
                best_val = combined
                save_checkpoint(
                    BEST_OUTPUT_PATH,
                    model,
                    optimizer,
                    step,
                    config,
                    best_val,
                    source_path.name,
                    source.get(
                        "source_step",
                        source.get("step"),
                    ),
                )
                print(
                    "  new best knowledge checkpoint saved"
                )

        if step % SAVE_EVERY == 0:
            save_checkpoint(
                OUTPUT_PATH,
                model,
                optimizer,
                step,
                config,
                best_val,
                source_path.name,
                source.get(
                    "source_step",
                    source.get("step"),
                ),
            )
            print("  resume checkpoint saved")

    save_checkpoint(
        OUTPUT_PATH,
        model,
        optimizer,
        STEPS,
        config,
        best_val,
        source_path.name,
        source.get(
            "source_step",
            source.get("step"),
        ),
    )

    print()
    print("=" * 72)
    print("Knowledge run complete")
    print("=" * 72)
    print(f"Latest: {OUTPUT_PATH}")
    print(f"Best:   {BEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
