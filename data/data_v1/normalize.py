"""Canonical document normalization. Sidecar is never this text."""

from __future__ import annotations

import re
import unicodedata

from data.data_v1 import PhaseBError
from data.data_v1.phase_a import document_id_for_text

URL_RE = re.compile(r"https?://", re.IGNORECASE)


def normalize_document(text: str) -> str:
    if not isinstance(text, str):
        raise PhaseBError("document text must be a string")
    out = unicodedata.normalize("NFKC", text)
    out = out.replace("\x00", "")
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    return out.strip()


def is_empty(text: str) -> bool:
    return not text.strip()


def document_id(text: str) -> str:
    return document_id_for_text(text)


def exact_hash(text: str) -> str:
    return document_id_for_text(text)


def contains_raw_url(text: str) -> bool:
    return bool(URL_RE.search(text or ""))
