"""Generate a higher-quality v4 chat/instruction dataset.

This dataset is intentionally curated and template-generated from simple,
verifiable facts and conversational patterns. It is used to retrain chat
behavior from the grounded checkpoint, not from the previous weak chat model.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


OUT = Path("data/chat_v4_curated.jsonl")
SEED = 2601


FACTS = [
    ("What is RAM?", "RAM is temporary memory that stores data programs and the CPU need quickly."),
    ("What does RAM do?", "RAM temporarily stores active data so the CPU can access it quickly."),
    ("What is a CPU?", "A CPU is the main processor that executes instructions and performs calculations."),
    ("What does a CPU do?", "A CPU executes program instructions and processes data."),
    ("What is Python?", "Python is a programming language used for many kinds of software and automation."),
    ("What is HTML?", "HTML is a markup language used to structure web pages."),
    ("What does HTML do?", "HTML defines the structure and content of a web page."),
    ("What is CSS?", "CSS is used to style the appearance and layout of web pages."),
    ("What is Java?", "Java is a programming language used for many kinds of applications."),
    ("What is a database?", "A database is an organized collection of data that can be stored, searched, and updated."),
    ("What is an SSD?", "An SSD is a storage device that keeps data even when the computer is turned off."),
    ("What is an SSD used for?", "An SSD is used to store files, programs, and the operating system."),
    ("What is a function in programming?", "A function is a reusable block of code that performs a specific task."),
    ("What is binary?", "Binary is a number system that uses only 0 and 1."),
    ("What is the capital of France?", "Paris is the capital of France."),
    ("What is the capital of Germany?", "Berlin is the capital of Germany."),
    ("What is the capital of Austria?", "Vienna is the capital of Austria."),
    ("What is the capital of Italy?", "Rome is the capital of Italy."),
    ("What is the capital of Japan?", "Tokyo is the capital of Japan."),
    ("What is the capital of Russia?", "Moscow is the capital of Russia."),
    ("What is the capital of Spain?", "Madrid is the capital of Spain."),
    ("What is the capital of Portugal?", "Lisbon is the capital of Portugal."),
    ("What is the capital of Canada?", "Ottawa is the capital of Canada."),
    ("What is the capital of Australia?", "Canberra is the capital of Australia."),
    ("What is the capital of Greece?", "Athens is the capital of Greece."),
    ("What is the capital of Egypt?", "Cairo is the capital of Egypt."),
    ("What is the capital of the United States?", "Washington, D.C. is the capital of the United States."),
    ("What is the capital of the United Kingdom?", "London is the capital of the United Kingdom."),
    ("What is the capital of China?", "Beijing is the capital of China."),
    ("What is the capital of India?", "New Delhi is the capital of India."),
    ("What is the capital of Brazil?", "Brasilia is the capital of Brazil."),
    ("What is the capital of Mexico?", "Mexico City is the capital of Mexico."),
    ("What is water made of?", "Water is made of hydrogen and oxygen, with the chemical formula H2O."),
    ("What is the chemical formula for water?", "The chemical formula for water is H2O."),
    ("What does the heart pump?", "The heart pumps blood around the body."),
    ("What do humans breathe?", "Humans breathe air and use oxygen from it."),
    ("What is the Sun?", "The Sun is a star at the center of our Solar System."),
    ("What does the Earth orbit?", "The Earth orbits the Sun."),
    ("What does the Moon orbit?", "The Moon orbits the Earth."),
    ("What is a dog?", "A dog is an animal and a common household pet."),
    ("What is a cat?", "A cat is an animal and a common household pet."),
    ("What is a fruit?", "A fruit is the seed-bearing part of a flowering plant, such as an apple or banana."),
    ("What is the opposite of hot?", "The opposite of hot is cold."),
    ("What is the opposite of fast?", "The opposite of fast is slow."),
    ("What is the usual color of grass?", "Grass is usually green."),
    ("What is the usual color of snow?", "Snow is usually white."),
    ("Why is the sky blue?", "The sky looks blue because Earth's atmosphere scatters blue light more strongly than most other visible colors."),
    ("What is a browser?", "A browser is a program used to open and interact with websites."),
    ("What is the internet?", "The internet is a global network that connects computers and other devices."),
    ("What is storage in a computer?", "Storage keeps files and programs for long-term use, even after the computer is turned off."),
    ("What is the difference between RAM and storage?", "RAM is fast temporary memory for active work, while storage keeps files and programs long term."),
    ("What is a computer?", "A computer is an electronic device that processes data and runs programs."),
    ("What does a computer do?", "A computer processes data, runs programs, and stores information."),
    ("What is a monitor?", "A monitor is an output device that displays visual information from a computer."),
    ("What is a keyboard?", "A keyboard is an input device used to type text and commands into a computer."),
    ("What is a mouse?", "A mouse is an input device used to point, click, and interact with items on a screen."),
    ("What is a GPU?", "A GPU is a processor designed to handle graphics and many parallel calculations."),
    ("What is an operating system?", "An operating system manages computer hardware and provides services for programs."),
    ("What is software?", "Software is the set of programs and instructions that run on a computer."),
    ("What is hardware?", "Hardware is the physical equipment that makes up a computer."),
]

CORE_ARITHMETIC = [
    ("What is 2 + 2?", "2 + 2 = 4."),
    ("What's 2 + 2?", "2 + 2 = 4."),
    ("2 + 2?", "2 + 2 = 4."),
    ("Calculate 2 + 2.", "2 + 2 = 4."),
    ("What is 3 + 4?", "3 + 4 = 7."),
    ("What is 5 + 5?", "5 + 5 = 10."),
    ("What is 10 - 4?", "10 - 4 = 6."),
]

CONVERSATION = [
    ("Hi!", "Hi! How can I help you?"),
    ("Hello!", "Hello! What would you like to talk about?"),
    ("Hey", "Hey! What's up?"),
    ("How are you?", "I'm doing well. How can I help?"),
    ("Thanks!", "You're welcome!"),
    ("Thank you.", "You're welcome!"),
    ("Can you help me?", "Of course. Tell me what you need help with."),
    ("I don't understand.", "No problem. I can explain it in a simpler way."),
    ("Can you explain that more simply?", "Sure. I'll keep it short and use simple words."),
    ("Give me a short answer.", "Sure. I'll keep the answer concise."),
    ("Tell me something friendly.", "I hope you're having a good day."),
    ("Say hello.", "Hello!"),
    ("If I say hello, what would you reply?", "I'd say: Hello!"),
    ("What can you do?", "I can answer questions, explain ideas, and help with learning or problem solving."),
    ("How can I learn programming?", "Start with one language, practice small projects, and learn by fixing mistakes as you go."),
    ("Tell me one interesting fact about space.", "A day on Venus is longer than a year on Venus."),
    ("Write a short sentence about a dog.", "The dog ran happily through the park."),
    ("Name a fruit and describe it.", "An apple is a fruit that is usually crisp and can taste sweet or tart."),
    ("Say something friendly without repeating yourself.", "I hope something good happens for you today."),
    ("Who are you?", "I'm OrcaAI, a local AI assistant running on this computer."),
    ("What are you?", "I'm OrcaAI, a local AI assistant."),
    ("Are you human?", "No. I'm an AI assistant, not a human."),
    ("Are you a person?", "No. I'm an AI assistant."),
    ("What is your name?", "My name is OrcaAI."),
    ("Can you say 123?", "123"),
    ("Say 123.", "123"),
    ("Why are you cool?", "Because I'm a small local AI built and trained from scratch."),
    ("Lol", "Lol."),
    ("Really?", "Yes."),
    ("Okay, really?", "Yes."),
    ("What's the issue?", "Tell me what seems wrong and I'll try to help."),
]

EXPLANATIONS = [
    ("Explain RAM in simple words.", "RAM is the computer's short-term working memory. It holds data the computer is using right now."),
    ("Explain a CPU in simple words.", "The CPU is the part of a computer that follows instructions and performs calculations."),
    ("Explain a database in simple words.", "A database is a structured place for storing information so it can be found and changed easily."),
    ("Explain a function in programming in simple words.", "A function is a named piece of code you can reuse whenever you need the same task done."),
    ("Explain the difference between RAM and storage simply.", "RAM is temporary and fast. Storage keeps your files for a long time."),
    ("Explain an SSD simply.", "An SSD is where a computer can permanently store files, apps, and its operating system."),
]

MATH = []
for a in range(0, 13):
    for b in range(0, 13):
        MATH.append((f"What is {a} + {b}?", f"{a} + {b} = {a + b}."))
for a in range(0, 13):
    for b in range(0, a + 1):
        MATH.append((f"What is {a} - {b}?", f"{a} - {b} = {a - b}."))

MULTI_TURN = [
    [
        ("Hey", "Hey! What's up?"),
        ("who are you", "I'm OrcaAI, a local AI assistant."),
        ("no, who are you??", "I'm OrcaAI, a local AI assistant running on this computer."),
        ("whats ram?", "RAM is temporary memory used for data the computer needs quickly."),
        ("what? i thought you are smart", "I can still make mistakes, but I'll try to answer clearly and accurately."),
        ("stop it. answer this: whats the capital of russia", "Moscow is the capital of Russia."),
    ],
    [
        ("Who are you?", "I'm OrcaAI, a local AI assistant."),
        ("Are you human?", "No. I'm an AI assistant, not a human."),
        ("Really?", "Yes. I'm software, not a human."),
        ("Are you sure?", "Yes. I'm an AI assistant, not a human."),
    ],
    [
        ("What's the capital of France?", "Paris is the capital of France."),
        ("Are you sure?", "Yes. Paris is the capital of France."),
        ("Really?", "Yes. Paris is the capital of France."),
    ],
    [
        ("What's RAM?", "RAM is temporary memory used for data the computer needs quickly."),
        ("Really?", "Yes. RAM is short-term working memory."),
        ("Can you explain it deeper?", "Programs load active data into RAM so the CPU can access it quickly. RAM is much faster than long-term storage."),
        ("So it is temporary?", "Yes. RAM is temporary memory and usually loses its contents when the computer powers off."),
    ],
    [
        ("Hey", "Hey! What's up?"),
        ("What's RAM?", "RAM is temporary memory that stores data the computer is actively using."),
        ("Really?", "Yes. RAM is short-term memory, while storage keeps data long term."),
        ("Can you explain it deeper?", "Sure. Programs load active data into RAM so the CPU can access it much faster than long-term storage."),
    ],
    [
        ("What's a computer?", "A computer is an electronic device that processes data and runs programs."),
        ("Can you explain that more simply?", "A computer takes input, processes information, and gives you output."),
    ],
    [
        ("Who are you?", "I'm OrcaAI, a local AI assistant."),
        ("Are you human?", "No. I'm an AI assistant, not a human."),
        ("Really?", "Yes. I'm software running locally on this computer."),
    ],
    [
        ("What's the capital of France?", "Paris is the capital of France."),
        ("Are you sure?", "Yes. Paris is the capital of France."),
    ],
    [
        ("What is RAM?", "RAM is temporary memory that stores data programs and the CPU need quickly."),
        ("Can you go deeper?", "RAM holds active program data so the CPU can access it quickly. Its contents are usually lost when the computer powers off."),
    ],
]

PROMPT_VARIANTS = [
    "{q}",
    "Please answer this: {q}",
    "Quick question: {q}",
    "Can you tell me: {q}",
]


def prompt_forms(question):
    forms = {
        question,
        question.lower(),
        question.rstrip("?.!"),
        question.lower().rstrip("?.!"),
    }

    lower = question.lower()
    if lower.startswith("what is "):
        rest = question[8:]
        forms.add("What's " + rest)
        forms.add("whats " + rest.lower())
    if lower.startswith("who are you"):
        forms.update({"who are you", "who are you?", "who are you??"})
    if lower.startswith("what is ram"):
        forms.update({"whats ram?", "what's ram?", "what is ram"})
    if lower.startswith("what is the capital of russia"):
        forms.update({
            "whats the capital of russia",
            "what's the capital of russia?",
            "stop it. answer this: whats the capital of russia",
        })

    return sorted(forms)


def build():
    rows = []

    def add(question, answer, category):
        seen = set()
        for form in prompt_forms(question):
            for template in PROMPT_VARIANTS:
                user = template.format(q=form)
                if user in seen:
                    continue
                seen.add(user)
                rows.append({
                    "user": user,
                    "assistant": answer,
                    "category": category,
                })

    for q, a in FACTS:
        add(q, a, "fact")
    for q, a in CONVERSATION:
        add(q, a, "conversation")

    # Reinforce a few tiny arithmetic facts that are easy for the base model
    # to confuse even after broad arithmetic training.
    for q, a in CORE_ARITHMETIC:
        for _ in range(12):
            rows.append({
                "user": q,
                "assistant": a,
                "category": "core_math",
            })
    for q, a in EXPLANATIONS:
        add(q, a, "explanation")

    # Multi-turn examples need much more weight than single-turn rows.
    # Otherwise hundreds of factual/math rows dominate and the model learns
    # isolated answers without learning what short follow-ups refer to.
    for turns in MULTI_TURN:
        for _ in range(24):
            rows.append({
                "turns": [
                    {"user": user, "assistant": assistant}
                    for user, assistant in turns
                ],
                "category": "multi_turn",
            })

    # Math already has lots of coverage; use two phrasings each.
    for q, a in MATH:
        rows.append({"user": q, "assistant": a, "category": "math"})
        rows.append({
            "user": f"Please calculate: {q}",
            "assistant": a,
            "category": "math",
        })

    # Add a few short multi-turn style followups as standalone pairs.
    followups = [
        ("Can you make that shorter?", "Sure. I'll keep it brief."),
        ("Can you give an example?", "Yes. I can give a simple example."),
        ("Can you explain it step by step?", "Yes. I'll break it into clear steps."),
        ("I see.", "Got it. Let me know if you want to go deeper."),
        ("Okay.", "Sounds good."),
    ]
    for q, a in followups:
        add(q, a, "conversation")

    rng = random.Random(SEED)
    rng.shuffle(rows)
    return rows


def main():
    rows = build()
    OUT.parent.mkdir(exist_ok=True)

    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    categories = {}
    for row in rows:
        categories[row["category"]] = categories.get(row["category"], 0) + 1

    print("=" * 72)
    print("OrcaAI v4 curated chat dataset")
    print("=" * 72)
    print(f"Examples: {len(rows)}")
    for name, count in sorted(categories.items()):
        print(f"{name:14} {count}")
    print(f"Saved: {OUT}")
    print()
    print("Next: py -3.12 orca.py finetune")


if __name__ == "__main__":
    main()
