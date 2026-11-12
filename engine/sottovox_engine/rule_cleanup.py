"""Deterministic zero-RAM rule-based transcript cleaner for Hush.

Replaces multi-gigabyte local LLMs when latency or an 8-16 GB RAM budget
matters: removes filler hesitations, collapses stutters, normalizes whitespace
and punctuation, and handles simple spoken commands in < 1 ms.
"""

from __future__ import annotations

import re

# Filler phrases/tokens stripped only when surrounded by boundaries so legitimate
# words ("umbrella", "likewise", "actually") inside sentences stay intact.
_STANDALONE_FILLER_RE = re.compile(
    r"(?i)(?:^|(?<=[\s,;:\(\[\"']))"
    r"(?:u+m+|u+h+|e+r+m+|a+h+|h+m+m*|mm-hmm|you know|i mean)"
    r"(?:[,;\s]+|(?=[.!?]|$))"
)

# Immediate consecutive word stutters: "the the", "I I", "to to"
_STUTTER_RE = re.compile(r"\b([A-Za-z']+)\s+\1\b", re.IGNORECASE)

# Spoken formatting tokens ("new line", "next line", "new paragraph")
_NEW_PARAGRAPH_RE = re.compile(r"(?i)\s*,?\s*\b(?:new paragraph|next paragraph)\b[.!,]?\s*")
_NEW_LINE_RE = re.compile(r"(?i)\s*,?\s*\b(?:new line|next line)\b[.!,]?\s*")

_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_REPEATED_PUNCT_RE = re.compile(r"([,.;:!?])\1+")


def clean_transcript_rules(text: str, *, custom_terms: dict[str, str] | None = None) -> str:
    """Apply fast deterministic cleanup rules to raw ASR output."""
    cleaned = text.strip()
    if not cleaned:
        return ""

    cleaned = _NEW_PARAGRAPH_RE.sub("\n\n", cleaned)
    cleaned = _NEW_LINE_RE.sub("\n", cleaned)
    cleaned = _STANDALONE_FILLER_RE.sub(" ", cleaned)
    cleaned = _STUTTER_RE.sub(r"\1", cleaned)
    cleaned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = re.sub(r",+\s*([.!?])", r"\1", cleaned)
    cleaned = _REPEATED_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)

    # Clean up dangling commas/whitespace at line edges
    lines = [re.sub(r"^[,;:\s]+", "", line.strip()) for line in cleaned.splitlines()]
    cleaned = "\n".join(lines).strip()

    if not cleaned:
        return ""

    # Capitalize the first alphabetic character of each line and sentence start,
    # and normalize standalone lowercase English pronoun "i" ("i", "i'm", "i'll", "i've", "i'd").
    cleaned = re.sub(
        r"(^|[.!?]\s+|\n+)([a-z])",
        lambda m: f"{m.group(1)}{m.group(2).upper()}",
        cleaned,
    )
    cleaned = re.sub(r"\bi(?=(?:'(?:m|ll|ve|d))?\b)", "I", cleaned)

    if custom_terms:
        for wrong, right in custom_terms.items():
            cleaned = re.sub(rf"\b{re.escape(wrong)}\b", right, cleaned, flags=re.IGNORECASE)

    return cleaned
