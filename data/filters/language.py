"""Script / language identification without an external LID model.

fasttext lid.176.bin is used when present; otherwise unicode script histograms
are the production fallback for SHINRA's multilingual mix.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path

SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "latin": ((0x0041, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF)),
    "cyrillic": ((0x0400, 0x04FF), (0x0500, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)),
    "greek": ((0x0370, 0x03FF),),
    "arabic": ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)),
    "hebrew": ((0x0590, 0x05FF),),
    "devanagari": ((0x0900, 0x097F),),
    "cjk": ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x3040, 0x30FF), (0xAC00, 0xD7AF)),
    "thai": ((0x0E00, 0x0E7F),),
}

ALLOWED_SCRIPTS = {
    "latin",
    "cyrillic",
    "greek",
    "arabic",
    "hebrew",
    "devanagari",
    "cjk",
    "thai",
}

ALLOWED_FASTTEXT_LANGS = {
    "en",
    "ru",
    "de",
    "fr",
    "es",
    "it",
    "pt",
    "nl",
    "pl",
    "uk",
    "cs",
    "ro",
    "sv",
    "hu",
    "fi",
    "tr",
    "ar",
    "he",
    "hi",
    "zh",
    "ja",
    "ko",
    "th",
    "vi",
    "id",
    "code",
}


def script_histogram(text: str, limit: int = 4000) -> Counter[str]:
    counts: Counter[str] = Counter()
    sample = text[:limit]
    for ch in sample:
        cp = ord(ch)
        if ch.isspace() or ch.isdigit() or ch in "{}[]()<>;,:._-+*/\\=|#@":
            continue
        matched = False
        for name, ranges in SCRIPT_RANGES.items():
            for start, end in ranges:
                if start <= cp <= end:
                    counts[name] += 1
                    matched = True
                    break
            if matched:
                break
        if not matched and ch.isalpha():
            counts["other"] += 1
    return counts


def dominant_script(text: str) -> tuple[str, float]:
    hist = script_histogram(text)
    total = sum(hist.values()) or 1
    script, count = hist.most_common(1)[0] if hist else ("latin", 0)
    return script, count / total


@lru_cache(maxsize=1)
def _fasttext_model(model_path: str):
    import fasttext

    return fasttext.load_model(model_path)


def detect_language(text: str, fasttext_model_path: str | None = None) -> dict:
    script, conf = dominant_script(text)
    result = {
        "script": script,
        "script_confidence": round(conf, 4),
        "language": "und",
        "language_confidence": 0.0,
        "keep": script in ALLOWED_SCRIPTS and conf >= 0.55,
    }
    path = fasttext_model_path or str(Path.home() / ".cache" / "fasttext" / "lid.176.bin")
    if Path(path).exists():
        model = _fasttext_model(path)
        labels, scores = model.predict(text.replace("\n", " ")[:4000], k=1)
        lang = labels[0].replace("__label__", "")
        score = float(scores[0])
        result["language"] = lang
        result["language_confidence"] = score
        result["keep"] = lang in ALLOWED_FASTTEXT_LANGS and score >= 0.35
        if script == "latin" and lang not in ALLOWED_FASTTEXT_LANGS:
            result["keep"] = False
    else:
        result["language"] = {
            "latin": "en",
            "cyrillic": "ru",
            "cjk": "zh",
            "arabic": "ar",
            "hebrew": "he",
            "devanagari": "hi",
            "greek": "el",
            "thai": "th",
        }.get(script, "und")
        result["language_confidence"] = conf
    return result
