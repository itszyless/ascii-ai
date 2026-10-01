"""Pretrain OrcaAI v5 from random weights."""

from __future__ import annotations

from pathlib import Path
import math
import random
import time

import numpy as np
import torch

from model_v5 import OrcaAIv5
from tokenizer import LocalTokenizer


DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints")

FINEWEB_TRAIN = DATA_DIR / "fineweb_v5.bin"
FINEWEB_VAL = DATA_DIR / "fineweb_v5_val.bin"
WIKI_TRAIN = DATA_DIR / "wikipedia_v5.bin"
WIKI_VAL = DATA_DIR / "wikipedia_v5_val.bin"

TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer_v5.json"
MODEL_PATH = CHECKPOINT_DIR / "model_v5.pt"
BEST_MODEL_PATH = CHECKPOINT_DIR / "model_v5_best.pt"

CONTEXT_LENGTH = 1024
EMBEDDING_SIZE = 768
NUM_HEADS = 12
NUM_LAYERS = 16
FFN_HIDDEN_SIZE = 2048
DROPOUT = 0.0

MICRO_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
TRAINING_STEPS = 120_000

LEARNING_RATE = 2.5e-4
MIN_LEARNING_RATE = 1.5e-5
WARMUP_STEPS = 3_000
WEIGHT_DECAY = 0.1

FINEWEB_WEIGHT = 0.80
WIKI_WEIGHT = 0.20

PRINT_EVERY = 25
EVAL_EVERY = 500
EVAL_BATCHES = 16
SAVE_EVERY = 500
SAMPLE_EVERY = 500

SEED = 2404

SAMPLE_PROMPTS = [
    "RAM is used to",
    "The CPU executes",
    "A function in programming is",
    "Paris is the capital of",
    "Water freezes at",
    "A dog is an",
]


def load_tokens(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: py -3.12 prepare_v5.py"
        )
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data, device):
    max_start = len(data) - CONTEXT_LENGTH - 1
    if max_start < 0:
        raise ValueError("Prepared dataset is too small.")

    starts = torch.randint(
        0,
        max_start + 1,
        (MICRO_BATCH_SIZE,),
    )

    x = torch.stack([
        torch.from_numpy(
            np.array(
                data[i:i + CONTEXT_LENGTH],
                dtype=np.int64,
            )
        )
        for i in starts
    ])
    y = torch.stack([
        torch.from_numpy(
            np.array(
                data[i + 1:i + CONTEXT_LENGTH + 1],
                dtype=np.int64,
            )
        )
        for i in starts
    ])

    return x.to(device), y.to(device)


def choose_source(fineweb, wikipedia):
    return (
        fineweb
        if random.random() < FINEWEB_WEIGHT
        else wikipedia
    )


def learning_rate(step):
    if step <= WARMUP_STEPS:
        return LEARNING_RATE * step / WARMUP_STEPS

    progress = (step - WARMUP_STEPS) / max(
        TRAINING_STEPS - WARMUP_STEPS,
        1,
    )
    progress = min(max(progress, 0.0), 1.0)

    cosine = 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )

    return MIN_LEARNING_RATE + cosine * (
        LEARNING_RATE - MIN_LEARNING_RATE
    )


@torch.no_grad()
def evaluate_source(
    model,
    data,
    device,
    use_bf16,
    batches,
):
    model.eval()
    losses = []

    for _ in range(batches):
        x, y = get_batch(data, device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            _, loss = model(x, y)

        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def evaluate(
    model,
    fineweb,
    wikipedia,
    device,
    use_bf16,
):
    fw_loss = evaluate_source(
        model,
        fineweb,
        device,
        use_bf16,
        EVAL_BATCHES,
    )
    wiki_loss = evaluate_source(
        model,
        wikipedia,
        device,
        use_bf16,
        EVAL_BATCHES,
    )
    combined = (
        FINEWEB_WEIGHT * fw_loss
        + WIKI_WEIGHT * wiki_loss
    )
    return combined, fw_loss, wiki_loss


@torch.no_grad()
def sample_text(
    model,
    tokenizer,
    device,
    prompt,
):
    model.eval()

    ids = tokenizer.encode(prompt)
    tokens = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device,
    )

    generated = model.generate(
        tokens,
        max_new_tokens=70,
        temperature=0.40,
        top_k=40,
        top_p=0.90,
        repetition_penalty=1.12,
        stop_token_ids={tokenizer.eos_id},
    )

    continuation = tokenizer.decode(
        generated[0, len(ids):].tolist()
    ).replace("\n", " ")

    model.train()
    return continuation.strip()


def save_checkpoint(
    path,
    model,
    optimizer,
    step,
    config,
    best_val,
):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "config": config,
            "stage": "pretrain_v5",
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

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_bf16 = (
        device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    config = {
        "vocab_size": tokenizer.vocab_size,
        "context_length": CONTEXT_LENGTH,
        "embedding_size": EMBEDDING_SIZE,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "ffn_hidden_size": FFN_HIDDEN_SIZE,
        "dropout": DROPOUT,
    }

    model = OrcaAIv5(**config).to(device)
    model.set_gradient_checkpointing(True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
    )

    start_step = 0
    best_val = float("inf")

    if MODEL_PATH.exists():
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=False,
        )

        if (
            checkpoint.get("stage") == "pretrain_v5"
            and checkpoint.get("config") == config
        ):
            model.load_state_dict(
                checkpoint["model_state"]
            )
            optimizer.load_state_dict(
                checkpoint["optimizer_state"]
            )
            start_step = checkpoint.get("step", 0)
            best_val = checkpoint.get(
                "best_val",
                float("inf"),
            )
            print(
                f"Resuming OrcaAI v5 from "
                f"step {start_step:,}"
            )

    params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    tokens_per_step = (
        MICRO_BATCH_SIZE
        * CONTEXT_LENGTH
        * GRAD_ACCUM_STEPS
    )
    total_positions = (
        TRAINING_STEPS * tokens_per_step
    )

    print("=" * 72)
    print("OrcaAI v5 pretraining")
    print("=" * 72)
    print(f"Device:               {device}")
    if device.type == "cuda":
        print(
            f"GPU:                  "
            f"{torch.cuda.get_device_name(0)}"
        )
    print(
        f"BF16:                 "
        f"{'yes' if use_bf16 else 'no'}"
    )
    print(f"Parameters:           {params:,}")
    print(f"Vocabulary:           {tokenizer.vocab_size:,}")
    print(f"Layers:               {NUM_LAYERS}")
    print(f"Context:              {CONTEXT_LENGTH}")
    print(f"Micro batch:          {MICRO_BATCH_SIZE}")
    print(f"Grad accumulation:    {GRAD_ACCUM_STEPS}")
    print(f"Effective batch:      {MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"Training steps:       {TRAINING_STEPS:,}")
    print("Gradient checkpoint: yes")
    print(f"Token positions:      {total_positions:,}")
    print(
        f"Data mix:             "
        f"{FINEWEB_WEIGHT:.0%} FineWeb-Edu / "
        f"{WIKI_WEIGHT:.0%} Wikipedia"
    )
    print()

    started = time.time()
    model.train()

    for step in range(
        start_step + 1,
        TRAINING_STEPS + 1,
    ):
        lr = learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for _ in range(GRAD_ACCUM_STEPS):
            source = choose_source(
                fineweb_train,
                wiki_train,
            )
            x, y = get_batch(source, device)

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
            accumulated_loss += (
                loss.item() / GRAD_ACCUM_STEPS
            )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )
        optimizer.step()

        if (
            step == start_step + 1
            or step % PRINT_EVERY == 0
        ):
            elapsed = max(
                time.time() - started,
                1e-6,
            )
            completed = step - start_step
            steps_per_second = (
                completed / elapsed
            )
            tokens_per_second = (
                steps_per_second
                * tokens_per_step
            )
            remaining = (
                TRAINING_STEPS - step
            )
            eta_hours = (
                remaining
                / max(steps_per_second, 1e-9)
                / 3600
            )

            print(
                f"Step {step:>6}/{TRAINING_STEPS} | "
                f"train {accumulated_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"{tokens_per_second:,.0f} tok/s | "
                f"ETA {eta_hours:.2f}h"
            )

        if step % EVAL_EVERY == 0:
            (
                val_loss,
                fw_loss,
                wiki_loss,
            ) = evaluate(
                model,
                fineweb_val,
                wiki_val,
                device,
                use_bf16,
            )

            print(
                f"  validation: {val_loss:.4f} | "
                f"FineWeb {fw_loss:.4f} | "
                f"Wikipedia {wiki_loss:.4f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    BEST_MODEL_PATH,
                    model,
                    optimizer,
                    step,
                    config,
                    best_val,
                )
                print(
                    "  new best v5 checkpoint saved"
                )

        if step % SAMPLE_EVERY == 0:
            print("  language samples:")
            for prompt in SAMPLE_PROMPTS:
                continuation = sample_text(
                    model,
                    tokenizer,
                    device,
                    prompt,
                )
                print(
                    f"    {prompt!r} -> "
                    f"{continuation}"
                )

        if step % SAVE_EVERY == 0:
            save_checkpoint(
                MODEL_PATH,
                model,
                optimizer,
                step,
                config,
                best_val,
            )
            print("  resume checkpoint saved")

    save_checkpoint(
        MODEL_PATH,
        model,
        optimizer,
        TRAINING_STEPS,
        config,
        best_val,
    )

    print()
    print("=" * 72)
    print("OrcaAI v5 pretraining complete")
    print("=" * 72)
    print(f"Resume checkpoint: {MODEL_PATH}")
    print(f"Best checkpoint:   {BEST_MODEL_PATH}")
    print()
    print("Next: py -3.12 orca.py test-v5")
    print(
        "Do not run finetune_v5.py until the "
        "base is coherent and relevant."
    )


if __name__ == "__main__":
    main()
