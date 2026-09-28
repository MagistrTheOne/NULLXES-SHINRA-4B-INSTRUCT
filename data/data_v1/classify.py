"""Closed taxonomy only. Ingest may not invent domains or languages."""

from __future__ import annotations

from data.data_v1 import PhaseBError
from data.data_v1.phase_a import DOMAINS, LANGUAGES, SOURCE_TYPES, SPLITS
from data.filters.language import dominant_script, script_histogram

CODE_HINTS = ("def ", "class ", "import ", "function ", "return ", "#!/")
MATH_HINTS = ("theorem", "lemma", "integral", "equals", "доказательств", "уравнен")
STRUCT_HINTS = ("{", "}", "<item>", "yaml:", "json")


def detect_language(text: str) -> str:
    """en|ru|und from unicode script only. Does not load fasttext or touch the network."""
    hist = script_histogram(text)
    latin = hist.get("latin", 0)
    cyr = hist.get("cyrillic", 0)
    other = sum(v for k, v in hist.items() if k not in {"latin", "cyrillic"})
    alpha = latin + cyr + other
    if alpha == 0:
        return "und"
    if cyr >= latin and cyr / alpha >= 0.35:
        return "ru"
    if latin / alpha >= 0.35 and other / alpha <= 0.45:
        return "en"
    script, conf = dominant_script(text)
    if script == "cyrillic" and conf >= 0.4:
        return "ru"
    if script == "latin" and conf >= 0.4:
        return "en"
    return "und"


def resolve_language(text: str, declared: str | None) -> str:
    if declared is not None and declared not in LANGUAGES:
        raise PhaseBError(f"language not in {LANGUAGES}: {declared}")
    detected = detect_language(text)
    if declared in LANGUAGES:
        if detected in LANGUAGES and detected != declared:
            raise PhaseBError("langid_mismatch")
        return declared
    if detected not in LANGUAGES:
        raise PhaseBError("language must be en|ru")
    return detected


def resolve_domain(text: str, declared: str | None) -> str:
    if declared is not None:
        if declared not in DOMAINS:
            raise PhaseBError(f"domain not in frozen enum: {declared}")
        return declared
    folded = text.casefold()
    if any(h in folded for h in CODE_HINTS):
        return "code"
    if any(h in folded for h in MATH_HINTS):
        return "math"
    if folded.lstrip().startswith(("{", "[")) or any(h in folded for h in STRUCT_HINTS):
        return "structured"
    return "general"


def resolve_split(declared: str | None) -> str:
    split = declared or "train"
    if split not in SPLITS:
        raise PhaseBError("split must be train|validation (probes are not a corpus split)")
    return split


def resolve_source_type(declared: str | None) -> str:
    source_type = declared or "natural"
    if source_type not in SOURCE_TYPES:
        raise PhaseBError(f"source_type not in {SOURCE_TYPES}")
    return source_type
