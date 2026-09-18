"""Scheduling tests use only local fixtures and the user's reported counters."""
from __future__ import annotations

import hashlib
import itertools
import sqlite3
from copy import deepcopy

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data import build_pilot as builder
from data import clean
from data.pilot_schedule import quota_conflicts, rank_sources
from data.pilot_state import PilotStore, empty_state
from data.sources import PILOT_BUCKET_WEIGHTS, PILOT_LANGUAGE_QUOTAS, PILOT_MIX


def reported_state():
    state = empty_state()
    state.update(
        bucket_counts={"general": 49_998_423, "code": 24_991_782, "math_stem": 13_049},
        lang_counts={"en": 74_999_949, "ru": 3_305},
        source_counts={"fineweb_edu_10bt": 49_998_423, "stack_smol": 24_991_782, "openwebmath": 13_049},
        total_tokens=75_003_254,
    )
    return state


def caps(max_tokens=100_000_000):
    return ({k: int(max_tokens * v) for k, v in PILOT_BUCKET_WEIGHTS.items()},
            {k: int(max_tokens * v) for k, v in PILOT_LANGUAGE_QUOTAS.items()})


def test_reported_75m_state_selects_only_ru_de_fr_and_no_scan_of_en():
    state = reported_state()
    before = deepcopy(state)
    buckets, languages = caps()
    order, skipped, _ = rank_sources(PILOT_MIX, state, buckets, languages, 100_000_000, 2048, {}, {}, {})
    assert set(order) == {"wikipedia_ru", "wikipedia_de", "wikipedia_fr"}
    assert skipped["fineweb_edu_10bt"] == "bucket_tail"
    for source in ("python_edu", "stack_smol", "openwebmath"):
        assert skipped[source] == "language_cap_or_tail"
    assert state == before


def test_openwebmath_is_selected_when_core_closed_and_en_has_room():
    state = reported_state()
    state["bucket_counts"]["multilingual"] = 10_000_000
    buckets, languages = caps()
    # Explicit alternate budget only for this test, never a production override.
    languages["en"] = 95_000_000
    closed = {"general": "bucket_tail", "code": "bucket_document_boundary"}
    order, skipped, _ = rank_sources(PILOT_MIX, state, buckets, languages, 100_000_000, 2048, closed, {}, {})
    assert order == ["openwebmath"]
    assert "stack_smol" in skipped


def test_conflict_quantifies_unfillable_math_without_changing_caps():
    state = reported_state()
    buckets, languages = caps()
    before = deepcopy((buckets, languages, PILOT_MIX))
    conflict, = quota_conflicts(PILOT_MIX, state, buckets, languages,
                               {"general": "bucket_tail", "code": "bucket_document_boundary"}, 2048)
    assert conflict["buckets"] == {"math_stem": 14_986_951}
    assert conflict["primary_languages"] == ["en"]
    assert conflict["available_tokens"] == 51
    assert conflict["shortfall_tokens"] == 14_986_900
    assert (buckets, languages, PILOT_MIX) == before


def test_ranking_changes_with_bucket_and_language_deficits():
    buckets, languages = caps()
    state = empty_state()
    state["bucket_counts"] = {"general": 45_000_000, "code": 24_000_000,
                              "math_stem": 1_000_000, "multilingual": 9_000_000}
    order, _, _ = rank_sources(PILOT_MIX, state, buckets, languages, 100_000_000, 2048, {}, {}, {})
    assert order[0] == "openwebmath"
    state["bucket_counts"]["multilingual"] = 0
    state["lang_counts"]["ru"] = languages["ru"]
    order, skipped, _ = rank_sources(PILOT_MIX, state, buckets, languages, 100_000_000, 2048, {}, {}, {})
    assert order[:2] == ["wikipedia_de", "wikipedia_fr"]
    assert skipped["wikipedia_ru"] == "language_cap_or_tail"


def row(key, tokens=100, language="en", source="fineweb_edu_10bt", bucket="general"):
    # Distinct two-word texts also give deterministic, cheap MinHash fixtures.
    prefix = f"fixture{key} "
    text = prefix + "x" * (tokens * 4 - len(prefix))
    return {"id": str(key), "text": text, "n_chars": len(text), "n_words": 2,
            "language": language, "script": "cyrillic" if language == "ru" else "latin",
            "domain": "encyclopedia" if bucket == "multilingual" else "web",
            "source": source, "bucket": bucket}


def fake_clean(record, *, dedup, **kwargs):
    if record.get("reject") or dedup.is_duplicate(record["id"], record["text"])[0]:
        return None
    return dict(record)


def run(root, **kwargs):
    args = dict(output_dir=root / "output", tokenizer_corpus_dir=root / "corpus",
                max_tokens=100_000, max_disk_gb=1e9, shard_size=8,
                tokenizer_max_chars=1_000_000, cache_dir=None)
    args.update(kwargs)
    return builder.build_pilot(**args)


def test_existing_checkpoint_and_sqlite_preserved_while_multilingual_fills(tmp_path, monkeypatch):
    output, corpus_dir = tmp_path / "output", tmp_path / "corpus"
    output.mkdir()
    corpus_dir.mkdir()
    corpus = corpus_dir / "pilot_corpus.txt"
    originals = [row("general", 49_998),
                 row("code", 24_991, source="stack_smol", bucket="code"),
                 row("math_en", 10, source="openwebmath", bucket="math_stem"),
                 row("r", 3, "ru", "openwebmath", "math_stem")]
    path = output / "clean-00009.parquet"
    pq.write_table(pa.Table.from_pylist(originals), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    store = PilotStore(output, corpus, 1_000_000)
    try:
        store.restore(True)
        saved_index = store.db.execute("SELECT id, digest, signature FROM documents ORDER BY id").fetchall()
        # An extra table survives too: resume must not replace the SQLite file.
        store.db.execute("CREATE TABLE preservation_marker (value TEXT)")
        store.db.execute("INSERT INTO preservation_marker VALUES ('keep')")
        store.db.commit()
    finally:
        store.close()
    prefix = corpus.read_bytes()
    def forbidden_rebuild(*_):
        pytest.fail("Scheduling-only change must not rebuild a valid checkpoint/index")
    monkeypatch.setattr(PilotStore, "reconstruct", forbidden_rebuild)
    monkeypatch.setattr(clean, "clean_record", fake_clean)
    opened = []
    def sources(spec, _):
        language = spec.get("language")
        assert language in {"ru", "de", "fr"}, "Blocked English source was scanned"
        opened.append(language)
        for i in range(200):
            yield row(f"{language}{i}", 100, language, bucket="multilingual")
    monkeypatch.setattr(clean, "iter_source", sources)
    report = run(tmp_path, resume=True)
    assert report["resume"]["mode"] == "checkpoint"
    assert report["resume"]["initial_docs"] == 4
    assert report["next_shard_idx"] > 10
    assert set(opened) == {"ru", "de", "fr"}
    assert len(opened) == 3  # No stream restart during scheduler switches.
    assert report["buckets_tokens"] == {"general": 49_998, "code": 24_991,
                                        "math_stem": 13, "multilingual": 10_000}
    assert report["languages_tokens"]["en"] == 74_999
    assert all(report["sources_tokens"].get(name, 0) > 0 for name in ("wikipedia_ru", "wikipedia_de", "wikipedia_fr"))
    assert report["stop_reason"] == "quota_conflict"
    assert report["completion_status"] == "blocked_by_language_quotas"
    assert report["tokens_est"] == 85_002
    assert report["kept"] == 104
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert corpus.read_bytes().startswith(prefix)
    with sqlite3.connect(output / "pilot_index.sqlite3") as db:
        after = db.execute("SELECT id, digest, signature FROM documents ORDER BY id").fetchall()
        assert set(saved_index).issubset(after)
        assert db.execute("SELECT value FROM preservation_marker").fetchone() == ("keep",)
    ids = [r["id"] for file in output.glob("clean-*.parquet") for r in pq.read_table(file, columns=["id"]).to_pylist()]
    assert len(ids) == len(set(ids)) == 104


def test_rejected_source_yields_and_does_not_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "PILOT_BUCKET_WEIGHTS", {"general": 0.5, "multilingual": 0.5})
    monkeypatch.setattr(builder, "PILOT_LANGUAGE_QUOTAS", {"en": 0.5, "ru": 0.5})
    monkeypatch.setattr(builder, "PILOT_MIX", {
        "bad": {"bucket": "general", "weight": 0.5, "hf_id": "bad"},
        "wikipedia_ru": {"bucket": "multilingual", "weight": 0.5, "language": "ru", "hf_id": "good"},
    })
    monkeypatch.setattr(clean, "clean_record", fake_clean)
    scanned, starts = {"bad": 0}, []
    def sources(spec, _):
        starts.append(spec["hf_id"])
        if spec["hf_id"] == "bad":
            for _ in itertools.count():
                scanned["bad"] += 1
                yield {"reject": True}
        else:
            assert scanned["bad"] == 4096, "Wiki was starved until the bad source hit its 5000-row limit"
            yield row("ru", 100, "ru", bucket="multilingual")
    monkeypatch.setattr(clean, "iter_source", sources)
    report = run(tmp_path, max_source_docs=5000)
    assert starts == ["bad", "good"]
    assert report["kept"] == 1
    assert scanned["bad"] == 5000
    assert report["stop_reason"] == "source_scan_limit"
