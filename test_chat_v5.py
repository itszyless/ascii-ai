"""Quick sanity test for the OrcaAI v5 chat checkpoint.

Checks for the failure modes seen in earlier versions: generic-answer collapse,
repetition, empty replies, and failure to follow the User:/Assistant: format.
It does not modify model weights.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean

import torch

from model_v5 import OrcaAIv5
from tokenizer import LocalTokenizer


CHECKPOINT = Path("checkpoints/chat_model_v5.pt")
TOKENIZER = Path("checkpoints/tokenizer_v5.json")

PROMPTS = [
    "Hello! What can you help me with?",
    "Tell me only your name.",
    "Are you a real person?",
    "Explain RAM like I'm 10.",
    "Explain RAM in more technical detail.",
    "What does a CPU do?",
    "What is the difference between a country and its capital city?",
    "What is the capital of Germany?",
    "Why does ice melt when it gets warm?",
    "Correct this sentence: helo my frend how is you",
    "Rewrite this politely: give me the file now",
    "Count from 1 to 8 with spaces only.",
    "Say exactly banana and nothing else.",
    "Do we know for certain that aliens exist?",
    "What gives life meaning?",
    "How would I start learning Java?",
    "Write one short sentence about a cat.",
    "Was ist RAM? Erkläre es einfach.",
    "Was ist die Hauptstadt von Deutschland?",
    "Korrigiere den Satz: ich bin gestern in die schule gegangen",
    "Erkläre mir auf Deutsch, was eine Datenbank ist.",
    "Réponds en français: qu'est-ce qu'un ordinateur ?",
    "Responde en español: ¿qué es una CPU?",
    "What is 13 * 7?",
    "What is 24 + 39?",
]

MULTI_TURN_TESTS = [
    [
        "What is RAM?",
        "Can you make that simpler?",
        "What happens to it when I turn the computer off?",
    ],
    [
        "Who are you?",
        "Are you human?",
        "Why not?",
    ],
    [
        "Was ist die Hauptstadt von Österreich?",
        "Bist du sicher?",
        "Und Deutschland?",
    ],
    [
        "Explain what a database is.",
        "Now explain the same thing in German.",
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


def should_use_history(user_prompt):
    text = " ".join(str(user_prompt).lower().split())
    standalone_markers = (
        "who are you",
        "what are you",
        "your name",
        "how are you",
        "what is ",
        "what's ",
        "whats ",
        "who is ",
        "where is ",
        "when is ",
        "why is ",
        "how does ",
        "how do ",
        "say ",
        "just say",
        "tell me ",
        "answer this",
        "calculate",
    )
    if any(marker in text for marker in standalone_markers):
        return False

    followup_markers = (
        "really",
        "are you sure",
        "go deeper",
        "explain it",
        "explain that",
        "what do you mean",
        "why?",
        "how so",
        "that",
        "this",
        "i never said",
    )
    return (
        len(text.split()) <= 8
        and any(marker in text for marker in followup_markers)
    )


def build_history_prompt(history, tokenizer, context_length):
    current_user = history[-1][1]
    if len(history) == 1:
        ids = tokenizer.encode(
            f"User: {current_user}\nAssistant:"
        )
        return ids[-context_length:]

    recent = history[-5:]
    token_ids = []
    pending_user = None

    for role, content in recent[:-1]:
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            token_ids.extend(
                tokenizer.encode(
                    f"User: {pending_user}\nAssistant:"
                )
            )
            token_ids.extend(
                tokenizer.encode(
                    f" {content}",
                    add_eos=True,
                )
            )
            pending_user = None

    token_ids.extend(
        tokenizer.encode(
            f"User: {current_user}\nAssistant:"
        )
    )
    return token_ids[-context_length:]


@torch.inference_mode()
def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            "Missing checkpoints/chat_model_v5.pt. Run chat fine-tuning first."
        )

    if not TOKENIZER.exists():
        raise FileNotFoundError(
            "Missing checkpoints/tokenizer_v5.json."
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

    model = OrcaAIv5(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("=" * 76)
    print("OrcaAI v5 held-out chat sanity test")
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
        "Send me this output before switching the web UI to v5."
    )


if __name__ == "__main__":
    main()
