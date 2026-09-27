"""Pretrain LocalAI v2 from random weights on the larger prepared corpus."""

from pathlib import Path
import math
import time

import numpy as np
import torch

from model import LocalAI
from tokenizer import LocalTokenizer


DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "train_v2.bin"
VAL_PATH = DATA_DIR / "val_v2.bin"
CHECKPOINT_DIR = Path("checkpoints")
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer_v2.json"
MODEL_PATH = CHECKPOINT_DIR / "model_v2.pt"
BEST_MODEL_PATH = CHECKPOINT_DIR / "model_v2_best.pt"

# ~30M parameter class: still practical on an RTX 4060 Ti 8 GB with BF16.
CONTEXT_LENGTH = 256
EMBEDDING_SIZE = 512
NUM_HEADS = 8
NUM_LAYERS = 8
DROPOUT = 0.1

BATCH_SIZE = 8
TRAINING_STEPS = 20_000
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
PRINT_EVERY = 25
EVAL_EVERY = 250
EVAL_BATCHES = 20
SAVE_EVERY = 500
SAMPLE_EVERY = 500


def load_tokens(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: py -3.12 prepare_data.py"
        )
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data, device):
    max_start = len(data) - CONTEXT_LENGTH - 1
    if max_start < 0:
        raise ValueError("Prepared dataset is too small.")

    starts = torch.randint(0, max_start + 1, (BATCH_SIZE,))
    x = torch.stack([
        torch.from_numpy(np.array(data[i:i + CONTEXT_LENGTH], dtype=np.int64))
        for i in starts
    ])
    y = torch.stack([
        torch.from_numpy(np.array(data[i + 1:i + CONTEXT_LENGTH + 1], dtype=np.int64))
        for i in starts
    ])
    return x.to(device), y.to(device)


def learning_rate(step):
    progress = min(step / TRAINING_STEPS, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LEARNING_RATE + cosine * (LEARNING_RATE - MIN_LEARNING_RATE)


@torch.no_grad()
def evaluate(model, data, device, use_bf16):
    model.eval()
    losses = []
    for _ in range(EVAL_BATCHES):
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
def print_sample(model, tokenizer, device):
    model.eval()
    prompt = "The computer"
    ids = tokenizer.encode(prompt)
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    generated = model.generate(
        tokens,
        max_new_tokens=80,
        temperature=0.8,
        top_k=50,
        stop_token_ids={tokenizer.eos_id},
    )
    text = tokenizer.decode(generated[0].tolist()).replace("\n", " ")
    print(f"  sample: {text[:500]}")
    model.train()


def save_checkpoint(path, model, optimizer, step, config, best_val):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "config": config,
            "stage": "pretrain_v2",
            "best_val": best_val,
        },
        path,
    )


def main():
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            "Missing v2 tokenizer. Run: py -3.12 prepare_data.py"
        )

    train_data = load_tokens(TRAIN_PATH)
    val_data = load_tokens(VAL_PATH)
    tokenizer = LocalTokenizer.load(TOKENIZER_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    config = {
        "vocab_size": tokenizer.vocab_size,
        "context_length": CONTEXT_LENGTH,
        "embedding_size": EMBEDDING_SIZE,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
    }

    model = LocalAI(**config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.01
    )

    start_step = 0
    best_val = float("inf")
    if MODEL_PATH.exists():
        old = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        if old.get("config") == config and old.get("stage") == "pretrain_v2":
            model.load_state_dict(old["model_state"])
            if "optimizer_state" in old:
                optimizer.load_state_dict(old["optimizer_state"])
            start_step = old.get("step", 0)
            best_val = old.get("best_val", float("inf"))
            print(f"Resuming v2 checkpoint from step {start_step:,}")

    parameter_count = sum(p.numel() for p in model.parameters())

    print("=" * 66)
    print("LocalAI v2 pretraining")
    print("=" * 66)
    print(f"Device:       {device}")
    if device.type == "cuda":
        print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"BF16:         {'yes' if use_bf16 else 'no'}")
    print(f"Parameters:   {parameter_count:,}")
    print(f"Vocabulary:   {tokenizer.vocab_size:,}")
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens:   {len(val_data):,}")
    print(f"Context:      {CONTEXT_LENGTH}")
    print(f"Batch size:   {BATCH_SIZE}")
    print()

    model.train()
    started = time.time()

    for step in range(start_step + 1, TRAINING_STEPS + 1):
        lr = learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            _, loss = model(x, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == start_step + 1 or step % PRINT_EVERY == 0:
            elapsed = max(time.time() - started, 1e-6)
            completed = step - start_step
            rate = completed / elapsed
            tokens_per_second = rate * BATCH_SIZE * CONTEXT_LENGTH
            remaining = max(TRAINING_STEPS - step, 0)
            eta_minutes = remaining / max(rate, 1e-9) / 60
            print(
                f"Step {step:>6}/{TRAINING_STEPS} | "
                f"train {loss.item():.4f} | lr {lr:.2e} | "
                f"{tokens_per_second:,.0f} tok/s | ETA {eta_minutes:.1f}m"
            )

        if step % EVAL_EVERY == 0:
            val_loss = evaluate(model, val_data, device, use_bf16)
            print(f"  validation loss: {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    BEST_MODEL_PATH, model, optimizer, step, config, best_val
                )
                print("  new best base checkpoint saved")

        if step % SAMPLE_EVERY == 0:
            print_sample(model, tokenizer, device)

        if step % SAVE_EVERY == 0:
            save_checkpoint(MODEL_PATH, model, optimizer, step, config, best_val)
            print("  resume checkpoint saved")

    save_checkpoint(MODEL_PATH, model, optimizer, TRAINING_STEPS, config, best_val)
    print()
    print(f"Pretraining complete: {MODEL_PATH}")
    print(f"Best validation checkpoint: {BEST_MODEL_PATH}")
    print("Next: inspect the printed samples before running finetune.py")


if __name__ == "__main__":
    main()
