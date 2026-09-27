"""Tokenizer trained from scratch on OrcaAI's own pretraining corpus."""

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]


class LocalTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.token_to_id("<pad>")
        self.bos_id = tokenizer.token_to_id("<bos>")
        self.eos_id = tokenizer.token_to_id("<eos>")

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    @classmethod
    def train(cls, texts, vocab_size=4096):
        tokenizer = Tokenizer(BPE(unk_token=None))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=True,
        )

        tokenizer.train_from_iterator(texts, trainer=trainer)
        return cls(tokenizer)

    def encode(self, text, add_bos=False, add_eos=False):
        ids = self.tokenizer.encode(text).ids

        if add_bos:
            ids.insert(0, self.bos_id)

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, token_ids):
        return self.tokenizer.decode(
            [int(token) for token in token_ids],
            skip_special_tokens=True,
        )

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))

    @classmethod
    def load(cls, path):
        return cls(Tokenizer.from_file(str(path)))
