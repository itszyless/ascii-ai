"""Byte-level tokenizer.

Every UTF-8 byte is a token. Unlike the old character tokenizer, this has a
fixed vocabulary and can accept text that never appeared in the training set.
"""

import json
from pathlib import Path


class ByteTokenizer:
    PAD = 256
    BOS = 257
    EOS = 258

    def __init__(self):
        self.vocab_size = 259

    def encode(self, text, add_bos=False, add_eos=False):
        tokens = list(text.encode("utf-8"))

        if add_bos:
            tokens.insert(0, self.BOS)

        if add_eos:
            tokens.append(self.EOS)

        return tokens

    def decode(self, token_ids):
        byte_values = [
            int(token)
            for token in token_ids
            if 0 <= int(token) <= 255
        ]
        return bytes(byte_values).decode("utf-8", errors="replace")

    def save(self, path):
        Path(path).write_text(
            json.dumps(
                {
                    "type": "byte",
                    "vocab_size": self.vocab_size,
                    "pad": self.PAD,
                    "bos": self.BOS,
                    "eos": self.EOS,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        # The file is kept with checkpoints so future tokenizer versions can
        # identify how a model was trained.
        json.loads(Path(path).read_text(encoding="utf-8"))
        return cls()
