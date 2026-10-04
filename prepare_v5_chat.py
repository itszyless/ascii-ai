"""Build OrcaAI v5 instruction/chat data.

Uses human-authored public instruction/conversation data plus a deterministic
core curriculum for behaviors that a small local assistant must get right.
No pretrained model weights or teacher model are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

from datasets import load_dataset


OUT = Path("data/chat_v5.jsonl")
SEED = 2605

OASST_NAME = "OpenAssistant/oasst1"
DOLLY_NAME = "databricks/databricks-dolly-15k"

TARGET_LANGS = {
    "en", "de", "fr", "es", "it", "pt", "nl", "pl", "cs", "sv"
}


def norm(text):
    return " ".join(str(text or "").split()).strip()


def stable_split(group_key):
    digest = hashlib.sha1(
        str(group_key).encode("utf-8", errors="ignore")
    ).digest()
    # Stable 90/10 split by conversation/task group, not random row.
    return "val" if int.from_bytes(digest[:4], "big") % 10 == 0 else "train"


def write_row(rows, turns, source, group):
    clean = []
    for user, assistant in turns:
        user = norm(user)
        assistant = norm(assistant)
        if user and assistant:
            clean.append({
                "user": user,
                "assistant": assistant,
            })

    if not clean:
        return

    rows.append({
        "turns": clean,
        "source": source,
        "group": str(group),
        "split": stable_split(f"{source}:{group}"),
    })


def load_oasst(rows, max_examples):
    print("Loading OpenAssistant OASST1...")
    ds = load_dataset(OASST_NAME, split="train")
    items = [dict(row) for row in ds]

    by_id = {
        row.get("message_id"): row
        for row in items
        if row.get("message_id")
    }

    candidates = []
    for row in items:
        if row.get("role") != "assistant":
            continue
        if row.get("deleted"):
            continue
        if row.get("synthetic"):
            continue
        if row.get("lang") not in TARGET_LANGS:
            continue

        review = row.get("review_result")
        if review is False:
            continue

        rank = row.get("rank")
        if rank is not None:
            try:
                if int(rank) > 1:
                    continue
            except (TypeError, ValueError):
                pass

        parent = by_id.get(row.get("parent_id"))
        if not parent or parent.get("role") != "prompter":
            continue

        # Walk to the root and convert the path into user/assistant turns.
        chain = []
        cursor = row
        seen = set()

        while cursor and cursor.get("message_id") not in seen:
            message_id = cursor.get("message_id")
            if message_id:
                seen.add(message_id)
            chain.append(cursor)
            parent_id = cursor.get("parent_id")
            cursor = by_id.get(parent_id) if parent_id else None

        chain.reverse()

        turns = []
        pending_user = None
        for message in chain:
            text = norm(message.get("text"))
            if not text:
                continue

            role = message.get("role")
            if role == "prompter":
                pending_user = text
            elif role == "assistant" and pending_user:
                turns.append((pending_user, text))
                pending_user = None

        if not turns:
            continue

        # Keep recent context so examples comfortably fit 1024 tokens.
        turns = turns[-4:]
        tree = row.get("message_tree_id") or row.get("message_id")
        candidates.append((turns, tree))

    rng = random.Random(SEED)
    rng.shuffle(candidates)

    seen = set()
    added = 0
    for turns, tree in candidates:
        key = tuple(
            (norm(user).lower(), norm(assistant).lower())
            for user, assistant in turns
        )
        if key in seen:
            continue
        seen.add(key)
        write_row(rows, turns, "oasst1", tree)
        added += 1
        if added >= max_examples:
            break

    print(f"  OASST1 examples: {added:,}")


def load_dolly(rows, max_examples):
    print("Loading Databricks Dolly 15k...")
    ds = load_dataset(DOLLY_NAME, split="train")
    candidates = []

    for index, row in enumerate(ds):
        instruction = norm(row.get("instruction"))
        context = norm(row.get("context"))
        response = norm(row.get("response"))

        if not instruction or not response:
            continue

        user = instruction
        if context:
            user += f"\n\nContext:\n{context}"

        candidates.append((user, response, index))

    rng = random.Random(SEED + 1)
    rng.shuffle(candidates)

    added = 0
    for user, response, index in candidates[:max_examples]:
        group = hashlib.sha1(
            user.encode("utf-8", errors="ignore")
        ).hexdigest()[:16]
        write_row(
            rows,
            [(user, response)],
            "dolly",
            group,
        )
        added += 1

    print(f"  Dolly examples:  {added:,}")


def add_core_curriculum(rows):
    print("Building OrcaAI core curriculum...")
    before = len(rows)

    # Identity + ordinary conversation.
    identity = [
        ("Who are you?", "I'm OrcaAI, a local AI assistant running on this computer."),
        ("What is your name?", "My name is OrcaAI."),
        ("Are you human?", "No. I'm an AI assistant, not a human."),
        ("Bist du ein Mensch?", "Nein. Ich bin ein KI-Assistent, kein Mensch."),
        ("Wie heißt du?", "Ich heiße OrcaAI."),
        ("Wer bist du?", "Ich bin OrcaAI, ein lokaler KI-Assistent auf diesem Computer."),
        ("How are you?", "I'm doing well. How can I help?"),
        ("Wie geht es dir?", "Mir geht es gut. Wie kann ich dir helfen?"),
    ]
    for index, pair in enumerate(identity):
        write_row(rows, [pair], "core_identity", f"identity-{index}")

    # Important uncertainty/opinion behavior.
    uncertainty = [
        (
            "Do you think aliens are real?",
            "Extraterrestrial life may exist, but we do not currently have confirmed evidence of alien life."
        ),
        (
            "Are aliens definitely real?",
            "We do not have confirmed evidence that extraterrestrial life exists, so I cannot say that it is definitely real."
        ),
        (
            "Was ist der Sinn des Lebens?",
            "Darauf gibt es keine allgemein bewiesene Antwort. Menschen finden Sinn zum Beispiel in Beziehungen, Zielen, Lernen, Kreativität oder darin, anderen zu helfen."
        ),
        (
            "What is the real purpose of life?",
            "There is no single scientifically proven purpose of life. People find meaning in different things, such as relationships, goals, learning, creativity, or helping others."
        ),
    ]
    for index, pair in enumerate(uncertainty):
        write_row(rows, [pair], "core_uncertainty", f"uncertainty-{index}")

    # Strict output and counting.
    for n in range(1, 31):
        answer = " ".join(str(i) for i in range(1, n + 1))
        write_row(
            rows,
            [(f"Count to {n} with spaces in between.", answer)],
            "core_format",
            f"count-en-{n}",
        )
        write_row(
            rows,
            [(f"Zähle bis {n} mit Leerzeichen dazwischen.", answer)],
            "core_format",
            f"count-de-{n}",
        )

    strict_values = ["123", "hello", "OrcaAI", "yes", "no", "42"]
    for value in strict_values:
        write_row(
            rows,
            [(f'Say exactly "{value}" and nothing else.', value)],
            "core_format",
            f"exact-en-{value}",
        )
        write_row(
            rows,
            [(f'Sage genau "{value}" und sonst nichts.', value)],
            "core_format",
            f"exact-de-{value}",
        )

    # Grammar correction. The split key groups variants of the same corrected
    # sentence together so validation is not a near-duplicate of training.
    corrections = [
        ("Helo, mY nam is Oliver", "Hello, my name is Oliver."),
        ("i like programing alot", "I like programming a lot."),
        ("she dont know the answer", "She doesn't know the answer."),
        ("we was at school yesterday", "We were at school yesterday."),
        ("can you helps me with this", "Can you help me with this?"),
        ("ich heiße oliver und ich mag programieren", "Ich heiße Oliver und ich mag Programmieren."),
        ("das ist ein sehr gute idee", "Das ist eine sehr gute Idee."),
        ("ich habe gestern ein film gesehen", "Ich habe gestern einen Film gesehen."),
        ("kannst du mir helfen bitte", "Kannst du mir bitte helfen?"),
    ]
    for index, (bad, good) in enumerate(corrections):
        prompts = [
            f'Correct the grammar and spelling: "{bad}"',
            f'Fix this sentence: "{bad}"',
            f'Korrigiere diesen Satz: "{bad}"',
        ]
        for variant, prompt in enumerate(prompts):
            write_row(
                rows,
                [(prompt, good)],
                "core_grammar",
                f"grammar-{index}",
            )

    # Computer concepts at multiple difficulty levels.
    concepts = {
        "RAM": {
            "normal": "RAM is fast temporary memory that holds data programs are actively using.",
            "easy": "RAM is your computer's short-term memory. It keeps things the computer needs right now.",
            "deep": "RAM stores active program data so the CPU can access it quickly. It is much faster than long-term storage and normally loses its contents when power is removed.",
        },
        "CPU": {
            "normal": "A CPU executes program instructions and processes data.",
            "easy": "The CPU is the part of a computer that follows instructions and does calculations.",
            "deep": "A CPU fetches, decodes, and executes instructions while coordinating calculations and data movement inside the computer.",
        },
        "database": {
            "normal": "A database is an organized collection of data that can be stored, searched, and updated.",
            "easy": "A database is an organized place where information is stored so it can be found later.",
            "deep": "A database stores structured information and provides ways to query, update, organize, and protect that information.",
        },
    }
    for concept, answers in concepts.items():
        write_row(
            rows,
            [(f"What is {concept}?", answers["normal"])],
            "core_explain",
            f"{concept}-normal",
        )
        write_row(
            rows,
            [(f"Explain {concept} in easier words.", answers["easy"])],
            "core_explain",
            f"{concept}-easy",
        )
        write_row(
            rows,
            [(f"Explain {concept} more deeply.", answers["deep"])],
            "core_explain",
            f"{concept}-deep",
        )
        write_row(
            rows,
            [
                (f"What is {concept}?", answers["normal"]),
                ("Explain that easier.", answers["easy"]),
                ("Can you explain it deeper?", answers["deep"]),
            ],
            "core_explain",
            f"{concept}-multiturn",
        )

    # Small arithmetic curriculum. The website also has an exact local
    # calculator for arithmetic; these examples mainly teach response format.
    rng = random.Random(SEED + 2)
    for index in range(2500):
        op = rng.choice(["+", "-", "*"])
        if op == "*":
            a = rng.randint(0, 99)
            b = rng.randint(0, 99)
            result = a * b
        elif op == "+":
            a = rng.randint(0, 999)
            b = rng.randint(0, 999)
            result = a + b
        else:
            a = rng.randint(0, 999)
            b = rng.randint(0, a)
            result = a - b

        forms = [
            (f"What is {a} {op} {b}?", str(result)),
            (f"Calculate {a} {op} {b}.", str(result)),
            (f"Wie viel ist {a} {op} {b}?", str(result)),
        ]
        user, answer = forms[index % len(forms)]
        write_row(
            rows,
            [(user, answer)],
            "core_math",
            f"math-{a}-{op}-{b}",
        )

    # General facts and comparisons that previously exposed v4 failure modes.
    facts = [
        ("What is the capital of France?", "Paris is the capital of France."),
        ("What is the capital of Russia?", "Moscow is the capital of Russia."),
        ("What is the capital of Austria?", "Vienna is the capital of Austria."),
        ("Was ist die Hauptstadt von Österreich?", "Wien ist die Hauptstadt von Österreich."),
        ("What is the difference between France and Paris?", "France is a country, while Paris is a city and the capital of France."),
        ("What is water made of?", "Water is made of hydrogen and oxygen. Its chemical formula is H2O."),
        ("Why is the sky blue?", "The sky looks blue because Earth's atmosphere scatters shorter blue wavelengths of sunlight more strongly than longer red wavelengths."),
    ]
    for index, pair in enumerate(facts):
        write_row(rows, [pair], "core_facts", f"fact-{index}")

    # Topic-preserving follow-up conversations.
    conversations = [
        [
            ("What's RAM?", "RAM is your computer's short-term working memory."),
            ("Explain it easier.", "RAM keeps the things your computer needs right now."),
            ("Still too difficult.", "RAM is like a small desk for the computer: things it is using now stay on the desk."),
        ],
        [
            ("Who are you?", "I'm OrcaAI, a local AI assistant."),
            ("Are you human?", "No. I'm software, not a human."),
            ("Really?", "Yes. I'm an AI program running locally on this computer."),
        ],
        [
            ("What's the capital of France?", "Paris is the capital of France."),
            ("Are you sure?", "Yes. Paris is the capital of France."),
            ("And Austria?", "Vienna is the capital of Austria."),
        ],
        [
            ("Was ist RAM?", "RAM ist der schnelle Arbeitsspeicher eines Computers."),
            ("Einfacher bitte.", "RAM merkt sich kurz die Dinge, die dein Computer gerade benutzt."),
            ("Und wenn ich den PC ausschalte?", "Der Inhalt des RAM geht normalerweise verloren, wenn der Computer ausgeschaltet wird."),
        ],
    ]
    for index, turns in enumerate(conversations):
        write_row(
            rows,
            turns,
            "core_multiturn",
            f"conversation-{index}",
        )

    print(f"  Core examples:   {len(rows) - before:,}")


def dedupe(rows):
    result = []
    seen = set()

    for row in rows:
        key = tuple(
            (
                turn["user"].lower(),
                turn["assistant"].lower(),
            )
            for turn in row["turns"]
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(row)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oasst-max", type=int, default=28_000)
    parser.add_argument("--dolly-max", type=int, default=15_000)
    args = parser.parse_args()

    rows = []
    load_oasst(rows, args.oasst_max)
    load_dolly(rows, args.dolly_max)
    add_core_curriculum(rows)

    rows = dedupe(rows)
    rng = random.Random(SEED + 3)
    rng.shuffle(rows)

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    train = sum(row["split"] == "train" for row in rows)
    val = sum(row["split"] == "val" for row in rows)

    by_source = {}
    for row in rows:
        by_source[row["source"]] = (
            by_source.get(row["source"], 0) + 1
        )

    print()
    print("=" * 72)
    print("OrcaAI v5 chat dataset ready")
    print("=" * 72)
    print(f"Total:      {len(rows):,}")
    print(f"Train:      {train:,}")
    print(f"Validation: {val:,}")
    for source, count in sorted(by_source.items()):
        print(f"{source:20} {count:,}")
    print(f"Saved:      {OUT}")
    print()
    print("Next: py -3.12 orca.py finetune-v5")


if __name__ == "__main__":
    main()
