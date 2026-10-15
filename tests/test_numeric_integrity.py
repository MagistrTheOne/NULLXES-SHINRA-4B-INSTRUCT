from __future__ import annotations

import hashlib
import json

import pytest

from data.numeric_integrity import DataIntegrityError, clean_text, numeric_spans, prepare_local_corpus


@pytest.mark.parametrize("value", ["2026", "-3.14159", "1,234.50", "12/31/2026", "9.81 m/s²", "1e-9", "50%"])
def test_numeric_spans_survive_cleaning(value):
    text = f"Измерение {value}\r\nточно"
    assert clean_text(text).splitlines() == [f"Измерение {value}", "точно"]
    assert numeric_spans(text) == numeric_spans(clean_text(text))


def test_unicode_normalization_cannot_rewrite_numeric_values():
    text = "value １２３"
    with pytest.raises(DataIntegrityError):
        clean_text(text, normalization="NFKC")


def test_corpus_requires_real_provenance_and_separates_splits(tmp_path):
    raw = "The year is 2026 and temperature is -4.5 C."
    row = {"text": raw, "source_id": "fixture-source", "revision": "immutable-1", "license": "CC0", "language": "en",
           "raw_sha256": hashlib.sha256(raw.encode()).hexdigest()}
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest = prepare_local_corpus([source], tmp_path / "out", split_salt="audit-frozen-salt")
    assert manifest["counts"]["kept"] == 1
    assert len({manifest[k] for k in ("train_hash", "validation_hash", "blind_hash")}) == 3
    stored = [json.loads(line) for name in ("train", "validation", "blind")
              for line in (tmp_path / "out" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    assert stored[0]["numeric_spans"]
    assert stored[0]["source_id"] == "fixture-source"


def test_bad_record_is_never_silently_skipped(tmp_path):
    source = tmp_path / "bad.jsonl"
    source.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="malformed JSON"):
        prepare_local_corpus([source], tmp_path / "out", split_salt="audit-frozen-salt")


def test_near_duplicates_are_removed_and_recorded(tmp_path):
    base = " ".join(f"researchword{i}" for i in range(80))
    changed = base.replace("researchword40", "measurement40")
    source = tmp_path / "near.jsonl"
    rows = []
    for index, text in enumerate((base, changed)):
        rows.append({"text": text, "source_id": "fixture", "revision": "immutable-1", "license": "CC0", "language": "en",
                     "document_id": f"doc-{index}", "raw_sha256": hashlib.sha256(text.encode()).hexdigest()})
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = prepare_local_corpus([source], tmp_path / "out", split_salt="audit-frozen-salt")
    assert manifest["counts"]["near_duplicates"] == 1
    duplicate = json.loads((tmp_path / "out" / "duplicates.jsonl").read_text(encoding="utf-8"))
    assert duplicate["type"] == "near"
    assert manifest["files"]["train.jsonl"] == manifest["train_hash"]
