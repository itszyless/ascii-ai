"""Character-level tokenizer for the first LocalAI Transformer.

This deliberately starts simple: every unique character in the training
corpus becomes one token. A subword tokenizer can replace this later.
"""

import json
from pathlib import Path


class CharacterTokenizer:
    def __init__(self, characters):
        self.characters = list(characters)
        self.char_to_id = {char: i for i, char in enumerate(self.characters)}
        self.id_to_char = {i: char for i, char in enumerate(self.characters)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.characters)

    def encode(self, text):
        unknown = sorted(set(text) - set(self.char_to_id))
        if unknown:
            raise ValueError(f"Tokenizer does not know these characters: {unknown}")
        return [self.char_to_id[char] for char in text]

    def decode(self, token_ids):
        return "".join(self.id_to_char[int(token_id)] for token_id in token_ids)

    def save(self, path):
        Path(path).write_text(
            json.dumps(self.characters, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        characters = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(characters)
