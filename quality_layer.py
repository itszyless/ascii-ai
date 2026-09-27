"""Small inference-quality layer for OrcaAI chat.

This does not replace model training. It improves practical chat behavior by:
- retrieving relevant local memories/knowledge
- retrying obviously broken generations
- choosing the least repetitive, most relevant candidate
"""

from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MEMORY_PATH = DATA_DIR / "lab_memories.json"
KNOWLEDGE_PATH = DATA_DIR / "lab_knowledge.json"

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _load(path):
    try:
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, list) else []
    except Exception:
        pass
    return []


def _words(text):
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def _score_overlap(query_words, text):
    if not query_words:
        return 0
    text_words = set(_words(text))
    return len(set(query_words) & text_words)


def retrieve_context(query, max_items=3, max_chars=900):
    query_words = _words(query)
    candidates = []

    for item in _load(MEMORY_PATH):
        text = str(item.get("text") or "").strip()
        score = _score_overlap(query_words, text)
        if score:
            candidates.append((score + 2, f"Memory: {text}"))

    for item in _load(KNOWLEDGE_PATH):
        title = str(item.get("title") or "")
        content = str(item.get("content") or "")
        score = _score_overlap(query_words, f"{title} {content}")
        if score >= 2:
            snippet = content[:600].replace("\n", " ")
            candidates.append((score, f"Knowledge ({title}): {snippet}"))

    candidates.sort(key=lambda pair: pair[0], reverse=True)

    selected = []
    used = 0
    for _, text in candidates:
        if len(selected) >= max_items:
            break
        remaining = max_chars - used
        if remaining <= 0:
            break
        text = text[:remaining]
        selected.append(text)
        used += len(text)

    return selected


def response_quality(text, prompt):
    """Heuristic ranker for several samples from the same small model."""
    clean = (text or "").strip()
    if not clean or clean in {".", "...", "…"}:
        return -100.0

    words = _words(clean)
    if not words:
        return -100.0

    score = 0.0

    # Reward some substance, but not rambling.
    score += min(len(words), 35) * 0.04
    if 4 <= len(words) <= 80:
        score += 1.0

    # Penalize severe token/phrase repetition.
    unique_ratio = len(set(words)) / max(len(words), 1)
    score += unique_ratio * 2.0
    if unique_ratio < 0.45:
        score -= 3.0

    repeated_runs = 0
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            repeated_runs += 1
    score -= repeated_runs * 1.5

    # Relevant content words are a useful weak signal.
    prompt_words = {
        word for word in _words(prompt)
        if len(word) > 2 and word not in {
            "what", "whats", "how", "are", "you", "the", "and",
            "can", "could", "would", "please", "tell", "about"
        }
    }
    if prompt_words:
        overlap = len(prompt_words & set(words))
        score += min(overlap, 4) * 0.8

    # Penalize very short non-greeting answers.
    greetings = {"hi", "hey", "hello", "yo", "sup"}
    if len(words) <= 2 and not (set(_words(prompt)) & greetings):
        score -= 2.0

    weird = sum(
        1 for char in clean
        if not (char.isalnum() or char.isspace() or char in ".,!?;:'\"-()[]/%+")
    )
    if weird > max(3, len(clean) * 0.08):
        score -= 2.0

    return score
