"""Conservative assistant-only fine-tuning for LocalAI v2.

Fixes:
- true next-token targets
- assistant-only loss
- exact-conversation deduplication
- multi-turn support
- lower LR + shorter max run to reduce catastrophic forgetting
"""

from pathlib import Path
import random
import re
import time

import torch
import torch.nn.functional as F

from model import LocalAI
from tokenizer import LocalTokenizer


CHAT_PATH = Path("data/chat.txt")
MODEL_PATH = Path("checkpoints/model_v2_best.pt")
FALLBACK_MODEL_PATH = Path("checkpoints/model_v2.pt")
TOKENIZER_PATH = Path("checkpoints/tokenizer_v2.json")
OUTPUT_PATH = Path("checkpoints/chat_model_v2.pt")

MAX_STEPS = 400
BATCH_SIZE = 12
LEARNING_RATE = 5e-6
PRINT_EVERY = 20
EVAL_EVERY = 20
EVAL_BATCHES = 30
VALIDATION_FRACTION = 0.10
EARLY_STOP_PATIENCE = 4
MIN_IMPROVEMENT = 0.003
SEED = 1337


def parse_conversations(text):
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]
    turn_pattern = re.compile(
        r"User:\s*(.*?)\r?\nAssistant:\s*(.*?)(?=\r?\nUser:|\Z)",
        re.DOTALL,
    )

    conversations = []
    seen = set()

    for block in blocks:
        turns = []
        for match in turn_pattern.finditer(block):
            user = match.group(1).strip()
            assistant = match.group(2).strip()
            if user and assistant:
                turns.append((user, assistant))

        if turns:
            key = tuple(
                (u.strip().lower(), a.strip().lower())
                for u, a in turns
            )
            if key not in seen:
                seen.add(key)
                conversations.append(turns)

    if len(conversations) < 20:
        raise ValueError("Need at least 20 unique conversations in data/chat.txt.")

    return conversations


def split_conversations(conversations):
    conversations = list(conversations)
    rng = random.Random(SEED)
    rng.shuffle(conversations)
    val_count = max(1, int(len(conversations) * VALIDATION_FRACTION))
    return conversations[val_count:], conversations[:val_count]


def make_example(turns, tokenizer, context_length):
    token_ids = []
    target_mask = []

    for user, assistant in turns:
        prompt_ids = tokenizer.encode(f"User: {user}\nAssistant:")
        response_ids = tokenizer.encode(f" {assistant}", add_eos=True)

        token_ids.extend(prompt_ids)
        target_mask.extend([False] * len(prompt_ids))
        token_ids.extend(response_ids)
        target_mask.extend([True] * len(response_ids))

    if len(token_ids) < 2:
        return None

    if len(token_ids) > context_length + 1:
        token_ids = token_ids[-(context_length + 1):]
        target_mask = target_mask[-(context_length + 1):]

    x_ids = token_ids[:-1]
    next_ids = token_ids[1:]
    next_mask = target_mask[1:]
    labels = [
        token if train_target else -100
        for token, train_target in zip(next_ids, next_mask)
    ]

    if all(label == -100 for label in labels):
        return None

    return (
        torch.tensor(x_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def build_examples(conversations, tokenizer, context_length):
    examples = []
    for conversation in conversations:
        example = make_example(conversation, tokenizer, context_length)
        if example is not None:
            examples.append(example)
    if not examples:
        raise ValueError("No usable chat examples after tokenization.")
    return examples


def get_batch(examples, tokenizer, context_length, device):
    chosen = random.choices(examples, k=BATCH_SIZE)
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


def assistant_loss(model, x, labels):
    logits, _ = model(x)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def evaluate(model, examples, tokenizer, context_length, device, use_bf16):
    model.eval()
    losses = []
    for _ in range(EVAL_BATCHES):
        x, labels = get_batch(examples, tokenizer, context_length, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            loss = assistant_loss(model, x, labels)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_best(model, config, step, train_loss, val_loss):
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "stage": "chat_v2",
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        OUTPUT_PATH,
    )


def main():
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError("Run v2 prepare_data.py first.")
    if not CHAT_PATH.exists():
        raise FileNotFoundError("Missing data/chat.txt.")

    base_path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL_PATH
    if not base_path.exists():
        raise FileNotFoundError("Run v2 train.py first.")

    tokenizer = LocalTokenizer.load(TOKENIZER_PATH)
    checkpoint = torch.load(base_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    raw_text = CHAT_PATH.read_text(encoding="utf-8")
    raw_blocks = len([b for b in re.split(r"\n\s*\n+", raw_text) if b.strip()])
    conversations = parse_conversations(raw_text)
    removed = raw_blocks - len(conversations)

    train_conversations, val_conversations = split_conversations(conversations)
    train_examples = build_examples(
        train_conversations, tokenizer, config["context_length"]
    )
    val_examples = build_examples(
        val_conversations, tokenizer, config["context_length"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    model = LocalAI(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    print("=" * 66)
    print("LocalAI v2 conservative assistant fine-tuning")
    print("=" * 66)
    print(f"Base checkpoint:       {base_path.name}")
    print(f"Unique conversations:  {len(conversations):,}")
    print(f"Duplicates removed:    {max(removed, 0):,}")
    print(f"Train conversations:   {len(train_examples):,}")
    print(f"Val conversations:     {len(val_examples):,}")
    print(f"Context length:        {config['context_length']} tokens")
    print(f"Learning rate:         {LEARNING_RATE:.1e}")
    print(f"Max steps:             {MAX_STEPS}")
    print()

    best_val = float("inf")
    stale_checks = 0
    started = time.time()

    for step in range(1, MAX_STEPS + 1):
        x, labels = get_batch(
            train_examples, tokenizer, config["context_length"], device
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            loss = assistant_loss(model, x, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % PRINT_EVERY == 0:
            rate = step / max(time.time() - started, 1e-6)
            print(
                f"Step {step:>4}/{MAX_STEPS} | "
                f"train {loss.item():.4f} | {rate:.2f} steps/s"
            )

        if step % EVAL_EVERY == 0:
            val_loss = evaluate(
                model, val_examples, tokenizer,
                config["context_length"], device, use_bf16
            )
            print(f"  assistant validation loss: {val_loss:.4f}")

            if val_loss < best_val - MIN_IMPROVEMENT:
                best_val = val_loss
                stale_checks = 0
                save_best(model, config, step, loss.item(), val_loss)
                print("  new best chat checkpoint saved")
            else:
                stale_checks += 1
                print(
                    f"  no meaningful improvement "
                    f"({stale_checks}/{EARLY_STOP_PATIENCE})"
                )
                if stale_checks >= EARLY_STOP_PATIENCE:
                    print("  early stopping")
                    break

    print()
    print(f"Best assistant validation loss: {best_val:.4f}")
    print(f"Chat checkpoint: {OUTPUT_PATH}")
    print("Run: py -3.12 web_app.py")


if __name__ == "__main__":
    main()
