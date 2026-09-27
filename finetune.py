"""Continue a pretrained LocalAI checkpoint on conversational examples.

Put examples in data/chat.txt using:
User: ...
Assistant: ...

This stage teaches response format/style; pretraining teaches general language.
"""

from pathlib import Path
import time

import torch

from model import LocalAI
from tokenizer import LocalTokenizer


CHAT_PATH = Path("data/chat.txt")
MODEL_PATH = Path("checkpoints/model.pt")
TOKENIZER_PATH = Path("checkpoints/tokenizer.json")
OUTPUT_PATH = Path("checkpoints/chat_model.pt")

STEPS = 1000
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
PRINT_EVERY = 25


def get_batch(data, context_length, device):
    max_start = len(data) - context_length - 1
    if max_start < 0:
        raise ValueError(
            f"data/chat.txt needs more than {context_length + 1} tokens."
        )

    starts = torch.randint(0, max_start + 1, (BATCH_SIZE,))
    x = torch.stack([data[i : i + context_length] for i in starts])
    y = torch.stack([data[i + 1 : i + context_length + 1] for i in starts])
    return x.to(device), y.to(device)


def main():
    if not MODEL_PATH.exists() or not TOKENIZER_PATH.exists():
        raise FileNotFoundError("Pretrain first with train.py.")

    if not CHAT_PATH.exists():
        raise FileNotFoundError(
            "Create data/chat.txt with User:/Assistant: conversations first."
        )

    tokenizer = LocalTokenizer.load(TOKENIZER_PATH)
    text = CHAT_PATH.read_text(encoding="utf-8")
    encoded = torch.tensor(tokenizer.encode(text, add_eos=True), dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = LocalAI(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    print("=" * 62)
    print("LocalAI conversational fine-tuning")
    print("=" * 62)
    print(f"Device:      {device}")
    print(f"Chat tokens: {len(encoded):,}")
    print(f"Steps:       {STEPS:,}")
    print()

    model.train()
    started = time.time()

    for step in range(1, STEPS + 1):
        x, y = get_batch(encoded, config["context_length"], device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % PRINT_EVERY == 0:
            rate = step / (time.time() - started)
            print(
                f"Step {step:>4}/{STEPS} | Loss {loss.item():.4f} | "
                f"{rate:.2f} steps/s"
            )

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "stage": "chat",
        },
        OUTPUT_PATH,
    )

    print()
    print(f"Done. Chat checkpoint saved to {OUTPUT_PATH}")
    print("Run: py -3.12 chat.py")


if __name__ == "__main__":
    main()
