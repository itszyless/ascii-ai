"""Generate a small factual grounding dataset for OrcaAI v4.

This creates many paraphrased prompt/answer pairs for basic facts and concepts.
It is intentionally small and targeted; it does not replace pretraining.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


OUTPUT_PATH = Path("data/grounding_v4.jsonl")
SEED = 2404

FACTS = [
    ("capital of France", "Paris"),
    ("capital of Germany", "Berlin"),
    ("capital of Italy", "Rome"),
    ("capital of Austria", "Vienna"),
    ("capital of Japan", "Tokyo"),
    ("what the Moon orbits", "the Earth"),
    ("what the Earth orbits", "the Sun"),
    ("freezing point of water in Celsius", "0 degrees Celsius"),
    ("boiling point of water in Celsius", "100 degrees Celsius"),
    ("two plus two", "four"),
    ("five plus five", "ten"),
    ("number of sides in a triangle", "three"),
    ("number of days in a week", "seven"),
    ("number of months in a year", "twelve"),
    ("opposite of hot", "cold"),
    ("opposite of up", "down"),
    ("what a dog is", "an animal"),
    ("what a cat is", "an animal"),
    ("what an apple is", "a fruit"),
    ("what a carrot is", "a vegetable"),
    ("what RAM is used for", "temporarily storing data that programs and the CPU need quickly"),
    ("what a CPU does", "executes instructions"),
    ("what a keyboard is", "an input device"),
    ("what a monitor is", "an output device"),
    ("what Python is", "a programming language"),
    ("what HTML is used for", "creating web pages"),
    ("what a function in programming is", "a reusable block of code that performs a task"),
    ("digits used in binary", "0 and 1"),
    ("system the brain belongs to", "the nervous system"),
    ("what the heart pumps", "blood"),
    ("what plants use sunlight for", "photosynthesis"),
    ("gas humans breathe to survive", "oxygen"),
    ("chemical formula for water", "H2O"),
    ("planet closest to the Sun", "Mercury"),
    ("largest planet in the Solar System", "Jupiter"),
    ("what the ocean is mostly made of", "water"),
    ("what English is", "a language"),
    ("what a book is made of", "pages"),
    ("what a car usually has", "wheels"),
    ("what birds have", "feathers"),
    ("where fish live", "water"),
    ("usual color of grass", "green"),
    ("usual color of snow", "white"),
    ("what the Sun provides", "light and heat"),
    ("what a database stores", "data"),
    ("what an SSD is used for", "data storage"),
    ("what a web browser is used for", "viewing websites"),
    ("what the internet connects", "computers and networks"),
]

TEMPLATES = [
    ("Question: What is {fact}?\nAnswer:", " {answer}"),
    ("Q: What is {fact}?\nA:", " {answer}"),
    ("Explain {fact}.\nAnswer:", " {answer}."),
    ("The correct answer for {fact} is", " {answer}."),
    ("Complete this fact: {fact}:", " {answer}."),
    ("In simple terms, {fact} is", " {answer}."),
]


def main():
    random.seed(SEED)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    rows = []
    for fact, answer in FACTS:
        for template, answer_template in TEMPLATES:
            prompt = template.format(fact=fact)
            response = answer_template.format(answer=answer)
            rows.append({
                "prompt": prompt,
                "response": response,
                "fact": fact,
            })

    random.shuffle(rows)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=" * 72)
    print("OrcaAI v4 grounding dataset")
    print("=" * 72)
    print(f"Facts:    {len(FACTS)}")
    print(f"Examples: {len(rows)}")
    print(f"Saved:    {OUTPUT_PATH}")
    print()
    print("Next: py -3.12 ground_v4.py")


if __name__ == "__main__":
    main()
