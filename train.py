"""Train LocalAI from random weights using data/training.txt."""

from pathlib import Path

import torch

from model import LocalAI
from tokenizer import CharacterTokenizer


DATA_PATH = Path("data/training.txt")
CHECKPOINT_DIR = Path("checkpoints")
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer.json"
MODEL_PATH = CHECKPOINT_DIR / "model.pt"

CONTEXT_LENGTH = 128
EMBEDDING_SIZE = 256
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT = 0.1

BATCH_SIZE = 32
TRAINING_STEPS = 5000
LEARNING_RATE = 3e-4
PRINT_EVERY = 100


def get_batch(data, device):
    max_start = len(data) - CONTEXT_LENGTH - 1

    if max_start < 0:
        raise ValueError(
            f"Training data is too short. It needs more than "
            f"{CONTEXT_LENGTH + 1} tokens."
        )

    starts = torch.randint(0, max_start + 1, (BATCH_SIZE,))

    x = torch.stack(
        [data[i : i + CONTEXT_LENGTH] for i in starts]
    )
    y = torch.stack(
        [data[i + 1 : i + CONTEXT_LENGTH + 1] for i in starts]
    )

    return x.to(device), y.to(device)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Missing data/training.txt. Add training text there before running train.py."
        )

    text = DATA_PATH.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError("data/training.txt is empty.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = CharacterTokenizer.from_text(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    tokenizer.save(TOKENIZER_PATH)

    model = LocalAI(
        vocab_size=tokenizer.vocab_size,
        context_length=CONTEXT_LENGTH,
        embedding_size=EMBEDDING_SIZE,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Training tokens: {len(encoded):,}")
    print(f"Parameters: {parameter_count:,}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()

    for step in range(1, TRAINING_STEPS + 1):
        x, y = get_batch(encoded, device)

        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % PRINT_EVERY == 0:
            print(
                f"Step {step:>5}/{TRAINING_STEPS} | "
                f"Loss: {loss.item():.4f}"
            )

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "context_length": CONTEXT_LENGTH,
                "embedding_size": EMBEDDING_SIZE,
                "num_heads": NUM_HEADS,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
            },
        },
        MODEL_PATH,
    )

    print()
    print(f"Training complete. Saved checkpoint to {MODEL_PATH}")


if __name__ == "__main__":
    main()
