"""Chat fine-tuning for a coherent OrcaAI v5 base model.

Do not run this until test_v5.py shows coherent, relevant continuations.
"""

from pathlib import Path
import json
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from model_v5 import OrcaAIv5
from tokenizer import LocalTokenizer


CHAT_PATH = Path("data/chat.txt")
CURATED_CHAT_PATH = Path("data/chat_v5.jsonl")
FINEWEB_PATH = Path("data/fineweb_v5.bin")
WIKI_PATH = Path("data/wikipedia_v5.bin")
MULTILINGUAL_PATH = Path("data/multilingual_v5_boost.bin")
BOOST_BEST_MODEL_PATH = Path("checkpoints/model_v5_boost_best.pt")
BOOST_MODEL_PATH = Path("checkpoints/model_v5_boost.pt")
MODEL_PATH = Path("checkpoints/model_v5_best.pt")
FALLBACK_MODEL_PATH = Path("checkpoints/model_v5.pt")
TOKENIZER_PATH = Path("checkpoints/tokenizer_v5.json")
OUTPUT_PATH = Path("checkpoints/chat_model_v5.pt")

MAX_STEPS = 6000
BATCH_SIZE = 1
LEARNING_RATE = 1.5e-5
PRINT_EVERY = 20
EVAL_EVERY = 100
EVAL_BATCHES = 48
VALIDATION_FRACTION = 0.10
EARLY_STOP_PATIENCE = 8
MIN_IMPROVEMENT = 0.002
BASE_REPLAY_EVERY = 4
BASE_REPLAY_WEIGHT = 0.08
MAX_SAME_RESPONSE = 3
SEED = 2404


def normalize(text):
    return " ".join(text.lower().split())



def load_curated_examples(path, split=None):
    if not path.exists():
        return []

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)

        row_split = item.get("split")
        if split is not None and row_split != split:
            continue

        turns = item.get("turns")
        if isinstance(turns, list):
            conversation = []
            for turn in turns:
                user = str(turn.get("user", "")).strip()
                assistant = str(turn.get("assistant", "")).strip()
                if user and assistant:
                    conversation.append((user, assistant))
            if conversation:
                rows.append(conversation)
            continue

        user = str(item.get("user", "")).strip()
        assistant = str(item.get("assistant", "")).strip()
        if user and assistant:
            rows.append([(user, assistant)])
    return rows


def parse_conversations(text):
    blocks = [
        block.strip()
        for block in re.split(r"\n\s*\n+", text)
        if block.strip()
    ]
    pattern = re.compile(
        r"User:\s*(.*?)\r?\nAssistant:\s*(.*?)(?=\r?\nUser:|\Z)",
        re.DOTALL,
    )

    conversations = []
    for block in blocks:
        turns = []
        for match in pattern.finditer(block):
            user = match.group(1).strip()
            assistant = match.group(2).strip()
            if user and assistant:
                turns.append((user, assistant))
        if turns:
            conversations.append(turns)

    if len(conversations) < 20:
        raise ValueError(
            "Need at least 20 conversations in data/chat.txt."
        )

    return conversations


def rebalance(conversations):
    unique_pairs = []
    seen_pairs = set()
    response_counts = {}

    for conversation in conversations:
        for user, assistant in conversation:
            user_key = normalize(user)
            answer_key = normalize(assistant)
            pair = (user_key, answer_key)

            if pair in seen_pairs:
                continue

            count = response_counts.get(answer_key, 0)
            if count >= MAX_SAME_RESPONSE:
                continue

            seen_pairs.add(pair)
            response_counts[answer_key] = count + 1
            unique_pairs.append([(user, assistant)])

    multi = []
    seen_multi = set()

    for conversation in conversations:
        if len(conversation) < 2:
            continue

        key = tuple(
            (normalize(user), normalize(assistant))
            for user, assistant in conversation
        )

        if key in seen_multi:
            continue

        seen_multi.add(key)
        multi.append(conversation)

    rng = random.Random(SEED)
    rng.shuffle(multi)

    multi_limit = min(
        len(multi),
        max(50, len(unique_pairs) // 2),
    )

    examples = unique_pairs + multi[:multi_limit]
    rng.shuffle(examples)
    return examples


def split_examples(examples):
    examples = list(examples)
    rng = random.Random(SEED)
    rng.shuffle(examples)

    count = max(
        1,
        int(len(examples) * VALIDATION_FRACTION),
    )
    return examples[count:], examples[:count]


def make_example(turns, tokenizer, context_length):
    token_ids = []
    target_mask = []

    for user, assistant in turns:
        prompt = tokenizer.encode(
            f"User: {user}\nAssistant:"
        )
        response = tokenizer.encode(
            f" {assistant}",
            add_eos=True,
        )

        token_ids.extend(prompt)
        target_mask.extend([False] * len(prompt))
        token_ids.extend(response)
        target_mask.extend([True] * len(response))

    if len(token_ids) < 2:
        return None

    if len(token_ids) > context_length + 1:
        token_ids = token_ids[-(context_length + 1):]
        target_mask = target_mask[-(context_length + 1):]

    x_ids = token_ids[:-1]
    next_ids = token_ids[1:]
    next_mask = target_mask[1:]

    labels = [
        token if enabled else -100
        for token, enabled in zip(
            next_ids,
            next_mask,
        )
    ]

    if all(label == -100 for label in labels):
        return None

    return (
        torch.tensor(x_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def build_examples(raw, tokenizer, context_length):
    result = []

    for turns in raw:
        item = make_example(
            turns,
            tokenizer,
            context_length,
        )
        if item is not None:
            result.append(item)

    if not result:
        raise ValueError("No usable chat examples.")

    return result


def get_chat_batch(
    examples,
    tokenizer,
    context_length,
    device,
):
    chosen = random.choices(
        examples,
        k=BATCH_SIZE,
    )

    batch_length = min(
        context_length,
        max(len(ids) for ids, _ in chosen),
    )
    x = torch.full(
        (BATCH_SIZE, batch_length),
        tokenizer.pad_id,
        dtype=torch.long,
    )
    labels = torch.full(
        (BATCH_SIZE, batch_length),
        -100,
        dtype=torch.long,
    )

    for row, (ids, row_labels) in enumerate(chosen):
        length = min(
            len(ids),
            batch_length,
        )
        x[row, :length] = ids[:length]
        labels[row, :length] = row_labels[:length]

    return x.to(device), labels.to(device)


def load_base_tokens(path):
    if not path.exists():
        return None
    return np.memmap(
        path,
        dtype=np.uint16,
        mode="r",
    )


def get_base_batch(
    data,
    context_length,
    device,
):
    if data is None:
        return None, None

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


def chat_loss(model, x, labels):
    logits, _ = model(x)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        label_smoothing=0.01,
    )


@torch.no_grad()
def evaluate_chat(
    model,
    examples,
    tokenizer,
    context_length,
    device,
    use_bf16,
):
    model.eval()
    losses = []

    for _ in range(EVAL_BATCHES):
        x, labels = get_chat_batch(
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
            loss = chat_loss(
                model,
                x,
                labels,
            )

        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def save_best(
    model,
    config,
    step,
    train_loss,
    val_loss,
):
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "stage": "chat_v5",
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        OUTPUT_PATH,
    )


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    if not CHAT_PATH.exists():
        raise FileNotFoundError(
            "Missing data/chat.txt."
        )

    model_path = (
        BOOST_BEST_MODEL_PATH
        if BOOST_BEST_MODEL_PATH.exists()
        else BOOST_MODEL_PATH
        if BOOST_MODEL_PATH.exists()
        else MODEL_PATH
        if MODEL_PATH.exists()
        else FALLBACK_MODEL_PATH
    )

    if (
        not model_path.exists()
        or not TOKENIZER_PATH.exists()
    ):
        raise FileNotFoundError(
            "Finish v5 pretraining first."
        )

    tokenizer = LocalTokenizer.load(
        TOKENIZER_PATH
    )
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    context_length = config["context_length"]

    curated_train = load_curated_examples(
        CURATED_CHAT_PATH,
        "train",
    )
    curated_val = load_curated_examples(
        CURATED_CHAT_PATH,
        "val",
    )

    if curated_train and curated_val:
        train_raw = curated_train
        val_raw = curated_val
        balanced = train_raw + val_raw
        rng = random.Random(SEED)
        rng.shuffle(train_raw)
        rng.shuffle(val_raw)
    else:
        conversations = parse_conversations(
            CHAT_PATH.read_text(
                encoding="utf-8"
            )
        )
        balanced = rebalance(conversations)
        train_raw, val_raw = split_examples(
            balanced
        )

    train_examples = build_examples(
        train_raw,
        tokenizer,
        context_length,
    )
    val_examples = build_examples(
        val_raw,
        tokenizer,
        context_length,
    )

    fineweb = load_base_tokens(FINEWEB_PATH)
    wikipedia = load_base_tokens(WIKI_PATH)
    multilingual = load_base_tokens(MULTILINGUAL_PATH)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_bf16 = (
        device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    model = OrcaAIv5(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.set_gradient_checkpointing(True)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    print("=" * 72)
    print("OrcaAI v5 chat fine-tuning")
    print("=" * 72)
    print(f"Base checkpoint:      {model_path.name}")
    print(
        f"Base step:            "
        f"{checkpoint.get('step', 'unknown')}"
    )
    print(
        f"Balanced examples:    "
        f"{len(balanced):,}"
    )
    print(
        f"Train examples:       "
        f"{len(train_examples):,}"
    )
    print(
        f"Validation examples:  "
        f"{len(val_examples):,}"
    )
    print(
        f"Parameters:           "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )
    print()

    best_val = float("inf")
    stale = 0
    started = time.time()

    for step in range(
        1,
        MAX_STEPS + 1,
    ):
        x, labels = get_chat_batch(
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
            loss = chat_loss(
                model,
                x,
                labels,
            )
            combined_loss = loss

            if step % BASE_REPLAY_EVERY == 0:
                roll = random.random()
                source = (
                    multilingual
                    if multilingual is not None and roll < 0.50
                    else fineweb
                    if fineweb is not None and roll < 0.85
                    else wikipedia
                )

                if source is not None:
                    base_x, base_y = get_base_batch(
                        source,
                        context_length,
                        device,
                    )
                    _, replay_loss = model(
                        base_x,
                        base_y,
                    )
                    combined_loss = (
                        loss
                        + BASE_REPLAY_WEIGHT
                        * replay_loss
                    )

        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )
        optimizer.step()

        if (
            step == 1
            or step % PRINT_EVERY == 0
        ):
            rate = step / max(
                time.time() - started,
                1e-6,
            )
            print(
                f"Step {step:>4}/{MAX_STEPS} | "
                f"chat {loss.item():.4f} | "
                f"{rate:.2f} steps/s"
            )

        if step % EVAL_EVERY == 0:
            val_loss = evaluate_chat(
                model,
                val_examples,
                tokenizer,
                context_length,
                device,
                use_bf16,
            )

            print(
                f"  validation loss: "
                f"{val_loss:.4f}"
            )

            if (
                val_loss
                < best_val - MIN_IMPROVEMENT
            ):
                best_val = val_loss
                stale = 0

                save_best(
                    model,
                    config,
                    step,
                    loss.item(),
                    val_loss,
                )
                print(
                    "  new best v5 chat "
                    "checkpoint saved"
                )
            else:
                stale += 1
                print(
                    f"  no meaningful improvement "
                    f"({stale}/{EARLY_STOP_PATIENCE})"
                )

                if stale >= EARLY_STOP_PATIENCE:
                    print("  early stopping")
                    break

    print()
    print(
        f"Best validation loss: "
        f"{best_val:.4f}"
    )
    print(f"Saved: {OUTPUT_PATH}")
    print("Next: py -3.12 orca.py test-chat-v5")


if __name__ == "__main__":
    main()
