"""Strict binary validation for SHINRA's SentencePiece vocab companion."""
from __future__ import annotations

import hashlib
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


EXPECTED_NATIVE_VOCAB_SIZE = 65_536
EXPECTED_NATIVE_VOCAB_SHA256 = "193cbf2e1dca090497352ba3b567e18740d0c63515a3ac9e886fc17db6562acc"
_SCORE_PATTERN = re.compile(rb"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _rounding_tolerance(score: Decimal) -> Decimal:
    """Return half of the least-significant written decimal unit, including exponents."""
    exponent = score.as_tuple().exponent
    return Decimal((0, (5,), exponent - 1))


def validate_sentencepiece_vocab(
    vocab_path: str | Path,
    sentencepiece_processor: Any,
    *,
    expected_vocab_size: int = EXPECTED_NATIVE_VOCAB_SIZE,
    expected_sha256: str = EXPECTED_NATIVE_VOCAB_SHA256,
) -> str:
    """Validate the exact model-ordered .vocab bytes and return their SHA-256.

    SentencePiece pieces may contain literal tabs, CRs, or LFs. Record boundaries
    are therefore found only after consuming the exact UTF-8 piece for each model ID.
    The score field remains numeric ASCII and its record separator is LF, CRLF, or CR.
    """
    path = Path(vocab_path)
    payload = path.read_bytes()
    actual_sha256 = _sha256(payload)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"SentencePiece vocab SHA256 mismatch: expected={expected_sha256}, actual={actual_sha256}"
        )

    piece_count = int(sentencepiece_processor.get_piece_size())
    if piece_count != expected_vocab_size:
        raise ValueError(
            f"SentencePiece model has {piece_count} pieces; expected exactly {expected_vocab_size}"
        )

    cursor = 0
    for token_id in range(piece_count):
        expected_piece = str(sentencepiece_processor.id_to_piece(token_id))
        expected_piece_bytes = expected_piece.encode("utf-8", errors="strict")
        expected_score = Decimal(str(sentencepiece_processor.get_score(token_id)))
        piece_end = cursor + len(expected_piece_bytes)
        actual_piece_bytes = payload[cursor:piece_end]
        if actual_piece_bytes != expected_piece_bytes:
            actual_piece = actual_piece_bytes.decode("utf-8", errors="replace")
            raise ValueError(
                f"SentencePiece vocab piece mismatch at ID {token_id}: "
                f"expected={expected_piece!r}, actual={actual_piece!r}"
            )
        cursor = piece_end

        if cursor >= len(payload) or payload[cursor] != 0x09:
            actual_byte = payload[cursor:cursor + 1]
            raise ValueError(
                f"SentencePiece vocab record mismatch at ID {token_id}: "
                f"expected tab after piece={expected_piece!r}, actual byte={actual_byte!r}"
            )
        cursor += 1

        lf = payload.find(b"\n", cursor)
        cr = payload.find(b"\r", cursor)
        terminators = [position for position in (lf, cr) if position >= 0]
        if terminators:
            record_end = min(terminators)
        elif token_id == piece_count - 1:
            record_end = len(payload)
        else:
            raise ValueError(
                f"SentencePiece vocab record mismatch at ID {token_id}: "
                "score record terminator is missing before the final ID"
            )

        score_bytes = payload[cursor:record_end]
        if not _SCORE_PATTERN.fullmatch(score_bytes):
            raise ValueError(
                f"SentencePiece vocab score mismatch at ID {token_id}: "
                f"expected piece={expected_piece!r}, score={str(expected_score)!r}; "
                f"actual score bytes={score_bytes!r}"
            )
        try:
            actual_score = Decimal(score_bytes.decode("ascii", errors="strict"))
        except (UnicodeDecodeError, InvalidOperation) as exc:
            raise ValueError(
                f"SentencePiece vocab score mismatch at ID {token_id}: "
                f"expected piece={expected_piece!r}, score={str(expected_score)!r}; "
                f"actual score bytes={score_bytes!r}"
            ) from exc
        if not actual_score.is_finite():
            raise ValueError(
                f"SentencePiece vocab score at ID {token_id} must be finite; "
                f"actual score={str(actual_score)!r}"
            )

        tolerance = _rounding_tolerance(actual_score)
        if abs(actual_score - expected_score) > tolerance:
            quantum = Decimal((0, (1,), actual_score.as_tuple().exponent))
            raise ValueError(
                f"SentencePiece vocab score mismatch at ID {token_id}: "
                f"expected piece={expected_piece!r}, score={str(expected_score)!r}; "
                f"actual piece={expected_piece!r}, score={str(actual_score)!r}; "
                f"written score quantum={quantum}, allowed absolute rounding error={tolerance} "
                "(half a written unit; exact boundary is accepted)"
            )

        if record_end == len(payload):
            cursor = record_end
        elif payload[record_end:record_end + 2] == b"\r\n":
            cursor = record_end + 2
        else:
            cursor = record_end + 1

    if cursor != len(payload):
        raise ValueError(
            f"SentencePiece vocab has trailing bytes after {piece_count} entries: "
            f"offset={cursor}, trailing_bytes={len(payload) - cursor}"
        )
    return actual_sha256
