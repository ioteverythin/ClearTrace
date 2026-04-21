"""Shared utility functions for Prism."""

from __future__ import annotations

import re
import hashlib
from typing import List, Tuple


def segment_sentences(text: str) -> List[str]:
    """Split text into sentences using regex-based heuristics.

    Handles common abbreviations and decimal numbers.
    """
    # Split on sentence-ending punctuation followed by whitespace + uppercase
    segments = re.split(r'(?<=[.!?])\s+(?=[A-Z"])', text)
    return [s.strip() for s in segments if s.strip()]


def tokenize_simple(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer.

    For real applications, users should use the model's own tokenizer.
    This is a fallback for provider-agnostic operation.
    """
    return re.findall(r"\b\w+\b|[^\w\s]", text)


def detokenize_simple(tokens: List[str]) -> str:
    """Reconstruct text from simple tokens."""
    result = []
    for i, token in enumerate(tokens):
        if i > 0 and re.match(r"\w", token) and re.match(r"\w", tokens[i - 1]):
            result.append(" ")
        elif i > 0 and token not in ".,!?;:)]}'\"" and tokens[i - 1] not in "([{'\"":
            result.append(" ")
        result.append(token)
    return "".join(result)


def cosine_similarity_text(text_a: str, text_b: str) -> float:
    """Compute a simple bag-of-words cosine similarity between two texts.

    Returns a value in [0.0, 1.0]. For production use, consider
    embedding-based similarity.
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / (len(words_a) ** 0.5 * len(words_b) ** 0.5)


def edit_distance(a: str, b: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return edit_distance(b, a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def text_hash(text: str) -> str:
    """Generate a short hash for a text string (for caching)."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def normalize_scores(scores: List[float]) -> List[float]:
    """Normalize a list of scores to [-1.0, 1.0] range."""
    if not scores:
        return []
    max_abs = max(abs(s) for s in scores)
    if max_abs == 0:
        return [0.0] * len(scores)
    return [s / max_abs for s in scores]


def diff_tokens(tokens_a: List[str], tokens_b: List[str]) -> List[Tuple[int, str, str]]:
    """Find positions where two token lists differ.

    Returns list of (position, old_token, new_token).
    """
    changes = []
    max_len = max(len(tokens_a), len(tokens_b))
    for i in range(max_len):
        tok_a = tokens_a[i] if i < len(tokens_a) else ""
        tok_b = tokens_b[i] if i < len(tokens_b) else ""
        if tok_a != tok_b:
            changes.append((i, tok_a, tok_b))
    return changes
