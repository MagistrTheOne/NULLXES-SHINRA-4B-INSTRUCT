"""Length, repetition, and structural quality filters (Gopher / FineWeb family)."""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass

WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")
ALPHA_RE = re.compile(r"[^\W\d_]", re.UNICODE)


@dataclass(frozen=True)
class QualityThresholds:
    min_chars: int = 200
    max_chars: int = 1_000_000
    min_words: int = 50
    max_words: int = 200_000
    min_mean_word_length: float = 3.0
    max_mean_word_length: float = 10.0
    max_symbol_ratio: float = 0.30
    max_ellipsis_ratio: float = 0.30
    min_alpha_ratio: float = 0.60
    max_duplicate_line_frac: float = 0.30
    max_duplicate_paragraph_frac: float = 0.30
    min_compression_ratio: float = 0.10
    max_compression_ratio: float = 0.65
    max_uppercase_frac: float = 0.40
    max_bullet_frac: float = 0.90


def word_list(text: str) -> list[str]:
    return WORD_RE.findall(text)


def mean_word_length(words: list[str]) -> float:
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def symbol_ratio(text: str) -> float:
    if not text:
        return 1.0
    symbols = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    return symbols / len(text)


def alpha_ratio(text: str) -> float:
    letters = ALPHA_RE.findall(text)
    return len(letters) / max(len(text), 1)


def uppercase_frac(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(ch.isupper() for ch in letters) / len(letters)


def duplicate_fraction(parts: list[str]) -> float:
    if not parts:
        return 0.0
    counts: dict[str, int] = {}
    for part in parts:
        key = part.strip()
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    dup = sum(c - 1 for c in counts.values() if c > 1)
    return dup / max(len(parts), 1)


def compression_ratio(text: str) -> float:
    raw = text.encode("utf-8", errors="ignore")
    if len(raw) < 64:
        return 0.5
    compressed = zlib.compress(raw, level=6)
    return len(compressed) / len(raw)


def score_document(text: str, thresholds: QualityThresholds | None = None) -> dict:
    cfg = thresholds or QualityThresholds()
    words = word_list(text)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    ellipsis = text.count("...") + text.count("…")
    bullets = sum(1 for ln in lines if ln.lstrip().startswith(("-", "*", "•", "·")))
    metrics = {
        "chars": len(text),
        "words": len(words),
        "mean_word_length": mean_word_length(words),
        "symbol_ratio": symbol_ratio(text),
        "alpha_ratio": alpha_ratio(text),
        "uppercase_frac": uppercase_frac(text),
        "ellipsis_ratio": ellipsis / max(len(lines), 1),
        "duplicate_line_frac": duplicate_fraction(lines),
        "duplicate_paragraph_frac": duplicate_fraction(paragraphs),
        "compression_ratio": compression_ratio(text),
        "bullet_frac": bullets / max(len(lines), 1),
    }
    reasons: list[str] = []
    if metrics["chars"] < cfg.min_chars or metrics["chars"] > cfg.max_chars:
        reasons.append("length_chars")
    if metrics["words"] < cfg.min_words or metrics["words"] > cfg.max_words:
        reasons.append("length_words")
    if not (cfg.min_mean_word_length <= metrics["mean_word_length"] <= cfg.max_mean_word_length):
        reasons.append("mean_word_length")
    if metrics["symbol_ratio"] > cfg.max_symbol_ratio:
        reasons.append("symbol_ratio")
    if metrics["alpha_ratio"] < cfg.min_alpha_ratio:
        reasons.append("alpha_ratio")
    if metrics["uppercase_frac"] > cfg.max_uppercase_frac:
        reasons.append("uppercase")
    if metrics["ellipsis_ratio"] > cfg.max_ellipsis_ratio:
        reasons.append("ellipsis")
    if metrics["duplicate_line_frac"] > cfg.max_duplicate_line_frac:
        reasons.append("dup_lines")
    if metrics["duplicate_paragraph_frac"] > cfg.max_duplicate_paragraph_frac:
        reasons.append("dup_paragraphs")
    if not (cfg.min_compression_ratio <= metrics["compression_ratio"] <= cfg.max_compression_ratio):
        reasons.append("compression")
    if metrics["bullet_frac"] > cfg.max_bullet_frac:
        reasons.append("bullets")
    metrics["keep"] = len(reasons) == 0
    metrics["drop_reasons"] = reasons
    metrics["quality_score"] = 0.0 if reasons else round(
        1.0
        - abs(metrics["mean_word_length"] - 5.0) / 10.0
        - metrics["symbol_ratio"]
        - metrics["duplicate_line_frac"] * 0.5,
        4,
    )
    return metrics
