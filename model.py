"""Decoder-only Transformer used by LocalAI."""

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
        self.qkv = nn.Linear(embedding_size, 3 * embedding_size, bias=False)
        self.projection = nn.Linear(embedding_size, embedding_size)
        self.dropout = dropout

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_size).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.projection(out)


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
        context_length=256,
        embedding_size=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(context_length, embedding_size)
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                embedding_size, num_heads, context_length, dropout
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embedding_size)
        self.output = nn.Linear(embedding_size, vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
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
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
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
    def generate(
        self,
        token_ids,
        max_new_tokens=200,
        temperature=0.35,
        top_k=20,
        stop_token_ids=None,
        repetition_penalty=1.08,
    ):
        self.eval()

        for _ in range(max_new_tokens):
            context = token_ids[:, -self.context_length:]
            logits, _ = self(context)
            logits = logits[:, -1, :]

            if repetition_penalty and repetition_penalty > 1.0:
                recent = token_ids[:, -64:]
                for batch_index in range(token_ids.size(0)):
                    used = torch.unique(recent[batch_index])
                    scores = logits[batch_index, used]
                    logits[batch_index, used] = torch.where(
                        scores < 0,
                        scores * repetition_penalty,
                        scores / repetition_penalty,
                    )

            logits = logits / max(temperature, 1e-5)

            if top_k:
                k = min(top_k, logits.size(-1))
                cutoff = torch.topk(logits, k).values[:, [-1]]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))

            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)

            if stop_token_ids and next_token.item() in stop_token_ids:
                break

        return token_ids
