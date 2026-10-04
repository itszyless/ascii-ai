"""OrcaAI v5 decoder-only Transformer.

v5 keeps the project small enough for a consumer GPU while moving closer to
modern decoder-only LLM design: RoPE, RMSNorm, SwiGLU, bias-free projections,
scaled residual initialization, tied embeddings, and SDPA attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dimension, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x_float = x.float()
        rms = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(rms + self.eps)
        return (x_norm.to(dtype) * self.weight)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_size, base=10_000.0):
        super().__init__()
        if head_size % 2 != 0:
            raise ValueError("RoPE head size must be even.")

        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, head_size, 2, dtype=torch.float32)
                / head_size
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, sequence_length, device):
        positions = torch.arange(
            sequence_length,
            device=device,
            dtype=torch.float32,
        )
        frequencies = torch.outer(positions, self.inv_freq)
        return frequencies.cos(), frequencies.sin()


def apply_rope(x, cos, sin):
    # x: [batch, heads, time, head_size]
    even = x[..., 0::2]
    odd = x[..., 1::2]
    cos = cos[None, None, :, :].to(dtype=x.dtype)
    sin = sin[None, None, :, :].to(dtype=x.dtype)

    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack(
        (rotated_even, rotated_odd),
        dim=-1,
    ).flatten(-2)


class CausalSelfAttentionV5(nn.Module):
    def __init__(self, embedding_size, num_heads, dropout):
        super().__init__()

        if embedding_size % num_heads != 0:
            raise ValueError(
                "embedding_size must be divisible by num_heads"
            )

        self.num_heads = num_heads
        self.head_size = embedding_size // num_heads
        self.dropout = dropout

        self.qkv = nn.Linear(
            embedding_size,
            3 * embedding_size,
            bias=False,
        )
        self.projection = nn.Linear(
            embedding_size,
            embedding_size,
            bias=False,
        )
        self.rope = RotaryEmbedding(self.head_size)

    def forward(self, x):
        batch, time, channels = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(
            batch, time, self.num_heads, self.head_size
        ).transpose(1, 2)
        k = k.view(
            batch, time, self.num_heads, self.head_size
        ).transpose(1, 2)
        v = v.view(
            batch, time, self.num_heads, self.head_size
        ).transpose(1, 2)

        cos, sin = self.rope(time, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(
            batch, time, channels
        )
        return self.projection(out)


class SwiGLU(nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout):
        super().__init__()
        self.gate = nn.Linear(
            embedding_size,
            hidden_size,
            bias=False,
        )
        self.up = nn.Linear(
            embedding_size,
            hidden_size,
            bias=False,
        )
        self.down = nn.Linear(
            hidden_size,
            embedding_size,
            bias=False,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.silu(self.gate(x)) * self.up(x)
        return self.dropout(self.down(x))


class TransformerBlockV5(nn.Module):
    def __init__(
        self,
        embedding_size,
        num_heads,
        hidden_size,
        dropout,
    ):
        super().__init__()
        self.norm1 = RMSNorm(embedding_size)
        self.attention = CausalSelfAttentionV5(
            embedding_size,
            num_heads,
            dropout,
        )
        self.norm2 = RMSNorm(embedding_size)
        self.feed_forward = SwiGLU(
            embedding_size,
            hidden_size,
            dropout,
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class OrcaAIv5(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_length=1024,
        embedding_size=768,
        num_heads=12,
        num_layers=16,
        ffn_hidden_size=2048,
        dropout=0.0,
    ):
        super().__init__()

        self.context_length = context_length
        self.embedding_size = embedding_size
        self.gradient_checkpointing = False

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_size,
        )
        self.blocks = nn.ModuleList([
            TransformerBlockV5(
                embedding_size,
                num_heads,
                ffn_hidden_size,
                dropout,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(embedding_size)
        self.output = nn.Linear(
            embedding_size,
            vocab_size,
            bias=False,
        )
        self.output.weight = self.token_embedding.weight

        self.apply(self._init_weights)

        # Scale residual branches so a deeper model starts stably.
        residual_std = 0.02 / math.sqrt(2 * num_layers)
        for block in self.blocks:
            nn.init.normal_(
                block.attention.projection.weight,
                mean=0.0,
                std=residual_std,
            )
            nn.init.normal_(
                block.feed_forward.down.weight,
                mean=0.0,
                std=residual_std,
            )

    def set_gradient_checkpointing(self, enabled=True):
        self.gradient_checkpointing = bool(enabled)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(self, token_ids, targets=None):
        _, sequence_length = token_ids.shape

        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"context length {self.context_length}"
            )

        x = self.token_embedding(token_ids)

        for block in self.blocks:
            if self.training and self.gradient_checkpointing:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

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
        temperature=0.45,
        top_k=40,
        top_p=0.92,
        stop_token_ids=None,
        repetition_penalty=1.10,
        repetition_window=96,
    ):
        self.eval()

        for _ in range(max_new_tokens):
            context = token_ids[:, -self.context_length:]
            logits, _ = self(context)
            logits = logits[:, -1, :]

            if repetition_penalty and repetition_penalty > 1.0:
                generated_so_far = token_ids[:, prompt_length:]
                if generated_so_far.numel() > 0:
                    recent = generated_so_far[:, -repetition_window:]
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
                cutoff = torch.topk(
                    logits,
                    k,
                ).values[:, [-1]]
                logits = logits.masked_fill(
                    logits < cutoff,
                    float("-inf"),
                )

            if top_p and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    logits,
                    descending=True,
                    dim=-1,
                )
                sorted_probs = F.softmax(
                    sorted_logits,
                    dim=-1,
                )
                cumulative = torch.cumsum(
                    sorted_probs,
                    dim=-1,
                )
                remove = cumulative > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits = sorted_logits.masked_fill(
                    remove,
                    float("-inf"),
                )
                logits = torch.full_like(
                    logits,
                    float("-inf"),
                ).scatter(
                    1,
                    sorted_indices,
                    sorted_logits,
                )

            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )
            token_ids = torch.cat(
                (token_ids, next_token),
                dim=1,
            )

            if (
                stop_token_ids
                and next_token.item() in stop_token_ids
            ):
                break

        return token_ids
