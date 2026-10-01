"""Quick sanity test for the OrcaAI v4 chat checkpoint.

Checks for the failure modes seen in earlier versions: generic-answer collapse,
repetition, empty replies, and failure to follow the User:/Assistant: format.
It does not modify model weights.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean

import torch

from model_v4 import OrcaAIv4
from tokenizer import LocalTokenizer


CHECKPOINT = Path("checkpoints/chat_model_v4.pt")
TOKENIZER = Path("checkpoints/tokenizer_v4.json")

PROMPTS = [
    "Hi! How are you?",
    "What is RAM?",
    "What does a CPU do?",
    "What is Python?",
    "What is 2 + 2?",
    "What is the capital of France?",
    "Why is the sky blue?",
    "Explain what a database is in simple words.",
    "What is the difference between RAM and storage?",
    "Tell me one interesting fact about space.",
    "How can I learn programming?",
    "Write a short sentence about a dog.",
    "What is an SSD used for?",
    "What does HTML do?",
    "If I say hello, what would you reply?",
    "Can you explain a function in programming?",
    "What is water made of?",
    "Name a fruit and describe it.",
    "What is the opposite of hot?",
    "Say something friendly without repeating yourself.",
]

MULTI_TURN_TESTS = [
    [
        "Hey",
        "Who are you?",
        "Are you human?",
        "Really?",
    ],
    [
        "What's RAM?",
        "Really?",
        "Can you explain it deeper?",
    ],
    [
        "What's the capital of France?",
        "Are you sure?",
    ],
]

TEMPERATURE = 0.20
TOP_K = 1
TOP_P = 1.0
REPETITION_PENALTY = 1.08
MAX_NEW_TOKENS = 100
SEED = 2404


def repetition_metrics(ids):
    if not ids:
        return 0.0, 0.0

    bigrams = list(zip(ids, ids[1:]))
    unique_bigram_ratio = (
        len(set(bigrams)) / len(bigrams)
        if bigrams
        else 1.0
    )

    counts = Counter(ids)
    max_token_fraction = max(counts.values()) / len(ids)
    return unique_bigram_ratio, max_token_fraction


def clean_reply(text):
    for marker in ("\nUser:", "\nAssistant:"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def build_history_prompt(history, tokenizer, context_length):
    recent = history[-7:]
    parts = []
    for role, content in recent:
        if role == "user":
            parts.append(f"User: {content}")
        else:
            parts.append(f"Assistant: {content}")
    parts.append("Assistant:")
    ids = tokenizer.encode("\n".join(parts))
    return ids[-context_length:]


@torch.inference_mode()
def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            "Missing checkpoints/chat_model_v4.pt. Run chat fine-tuning first."
        )

    if not TOKENIZER.exists():
        raise FileNotFoundError(
            "Missing checkpoints/tokenizer_v4.json."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = LocalTokenizer.load(TOKENIZER)
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    model = OrcaAIv4(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("=" * 76)
    print("OrcaAI v4 chat sanity test")
    print("=" * 76)
    print(f"Device:      {device}")
    if device.type == "cuda":
        print(f"GPU:         {torch.cuda.get_device_name(0)}")
    print(f"Chat step:   {checkpoint.get('step', 'unknown')}")
    print(f"Val loss:    {checkpoint.get('val_loss', 'unknown')}")
    print()

    replies = []
    lengths = []
    bigram_ratios = []
    max_token_fractions = []
    empty = 0

    for index, user_prompt in enumerate(PROMPTS, 1):
        prompt = f"User: {user_prompt}\nAssistant:"
        prompt_ids = tokenizer.encode(prompt)
        x = torch.tensor(
            [prompt_ids[-model.context_length:]],
            dtype=torch.long,
            device=device,
        )

        torch.manual_seed(SEED + index)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(SEED + index)

        generated = model.generate(
            x,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            stop_token_ids={tokenizer.eos_id},
        )

        new_ids = generated[0, x.size(1):].tolist()
        reply = clean_reply(tokenizer.decode(new_ids))

        if not reply:
            empty += 1

        bigram_ratio, max_fraction = repetition_metrics(new_ids)
        replies.append(reply)
        lengths.append(len(new_ids))
        bigram_ratios.append(bigram_ratio)
        max_token_fractions.append(max_fraction)

        print(f"[{index:02d}] User: {user_prompt}")
        print(f"     OrcaAI: {reply or '<EMPTY>'}")
        print()

    print("=" * 76)
    print("Multi-turn checks")
    print("=" * 76)

    for convo_index, turns in enumerate(MULTI_TURN_TESTS, 1):
        history = []
        print(f"Conversation {convo_index}:")
        for turn_index, user_prompt in enumerate(turns, 1):
            history.append(("user", user_prompt))
            prompt_ids = build_history_prompt(
                history,
                tokenizer,
                model.context_length,
            )
            x = torch.tensor(
                [prompt_ids],
                dtype=torch.long,
                device=device,
            )

            generated = model.generate(
                x,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                stop_token_ids={tokenizer.eos_id},
            )
            new_ids = generated[0, x.size(1):].tolist()
            reply = clean_reply(tokenizer.decode(new_ids))
            history.append(("assistant", reply))
            print(f"  User: {user_prompt}")
            print(f"  OrcaAI: {reply or '<EMPTY>'}")
        print()

    normalized = [
        " ".join(reply.lower().split())
        for reply in replies
        if reply.strip()
    ]
    duplicate_replies = (
        len(normalized) - len(set(normalized))
    )

    print("=" * 76)
    print("Summary")
    print("=" * 76)
    print(f"Prompts:               {len(PROMPTS)}")
    print(f"Empty replies:         {empty}")
    print(f"Exact duplicate replies:{duplicate_replies}")
    print(f"Mean output length:    {mean(lengths):.1f} tokens")
    print(
        f"Mean unique bigrams:   {mean(bigram_ratios):.3f}"
    )
    print(
        f"Mean max-token share:  {mean(max_token_fractions):.3f}"
    )
    print()
    print(
        "Send me this output before relying on the web UI."
    )


if __name__ == "__main__":
    main()
