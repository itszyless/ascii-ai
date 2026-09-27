"""A small decoder-only Transformer language model built with PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_size, num_heads, context_length, dropout):
        super().__init__()

        if embedding_size % num_heads != 0:
            raise ValueError("embedding_size must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_size = embedding_size // num_heads

        self.query = nn.Linear(embedding_size, embedding_size, bias=False)
        self.key = nn.Linear(embedding_size, embedding_size, bias=False)
        self.value = nn.Linear(embedding_size, embedding_size, bias=False)
        self.output = nn.Linear(embedding_size, embedding_size)

        self.dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(context_length, context_length))
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        batch_size, sequence_length, embedding_size = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(batch_size, sequence_length, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, sequence_length, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, sequence_length, self.num_heads, self.head_size).transpose(1, 2)

        attention = q @ k.transpose(-2, -1)
        attention = attention / (self.head_size ** 0.5)

        mask = self.causal_mask[:sequence_length, :sequence_length]
        attention = attention.masked_fill(mask == 0, float("-inf"))

        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        out = attention @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_size)

        return self.dropout(self.output(out))


class FeedForward(nn.Module):
    def __init__(self, embedding_size, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.GELU(),
            nn.Linear(4 * embedding_size, embedding_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, context_length, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_size)
        self.attention = CausalSelfAttention(
            embedding_size, num_heads, context_length, dropout
        )
        self.norm2 = nn.LayerNorm(embedding_size)
        self.feed_forward = FeedForward(embedding_size, dropout)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class LocalAI(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_length=128,
        embedding_size=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
    ):
        super().__init__()

        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(context_length, embedding_size)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_size,
                    num_heads,
                    context_length,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(embedding_size)
        self.output = nn.Linear(embedding_size, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, targets=None):
        _, sequence_length = token_ids.shape

        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds context length "
                f"{self.context_length}"
            )

        positions = torch.arange(sequence_length, device=token_ids.device)

        x = self.token_embedding(token_ids)
        x = x + self.position_embedding(positions)
        x = self.blocks(x)
        x = self.final_norm(x)

        logits = self.output(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, token_ids, max_new_tokens, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.context_length :]
            logits, _ = self(context)

            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                logits[logits < values[:, [-1]]] = float("-inf")

            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)

        return token_ids
