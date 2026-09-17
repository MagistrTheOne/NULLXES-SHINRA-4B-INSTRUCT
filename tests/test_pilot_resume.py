"""Local-only regression coverage; no Hugging Face datasets are downloaded."""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import itertools
import json
import logging
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data import build_pilot as builder
from data import clean
from data.pilot_state import (
    PilotStore, atomic_json, build_lock, estimate_tokens, publish_shard,
)


def document(i, chars=400, language="en"):
    prefix = f"document{i} "
    text = prefix + (chr(97 + int(hashlib.sha256(str(i).encode()).hexdigest(), 16) % 26) * (chars - len(prefix)))
    return dict(id=str(i), text=text, n_chars=len(text), n_words=2,
                language=language, script="latin", domain="web", quality_score=1.0,
                toxicity=0.0, code_language=None, dedup="keep",
                source="fineweb_edu_10bt", bucket="general")


def synthetic_clean(row, *, dedup, **kwargs):
    if row.get("reject"):
        return None
    row = dict(row)
    if dedup.is_duplicate(row["id"], row["text"])[0]:
        return None
    return row


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "PILOT_BUCKET_WEIGHTS", {"general": 1.0})
    monkeypatch.setattr(builder, "PILOT_LANGUAGE_QUOTAS", {"en": 1.0})
    monkeypatch.setattr(builder, "PILOT_MIX", {
        "fineweb_edu_10bt": {"bucket": "general", "domain": "web", "hf_id": "local"},
    })
    monkeypatch.setattr(clean, "clean_record", synthetic_clean)
    def run(rows=(), **kwargs):
        monkeypatch.setattr(clean, "iter_source", lambda *_: iter(rows))
        args = dict(output_dir=tmp_path / "output", tokenizer_corpus_dir=tmp_path / "corpus",
                    max_tokens=10_000, max_disk_gb=1e9, shard_size=2,
                    tokenizer_max_chars=1_000_000, cache_dir=None, tail_tokens=0,
                    max_source_docs=100)
        args.update(kwargs)
        return builder.build_pilot(**args)
    return run


def parquet_rows(path):
    return [row for file in sorted(path.glob("clean-*.parquet")) for row in pq.read_table(file).to_pylist()]


def snapshot(path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.glob("clean-*.parquet")}


def test_fresh_empty(setup, tmp_path):
    report = setup()
    assert report["tokens_est"] == report["kept"] == report["shards"] == 0
    assert report["resume"]["mode"] == "fresh"
    assert report["stop_reason"] == "sources_exhausted"
    assert report["tokenizer_friendship"]["dna"]["vocab_size_target"] == 131072
    assert json.loads((tmp_path / "output/pilot_state.json").read_text())["written_docs"] == 0


def test_fresh_build_report_and_progress(setup, capsys):
    report = setup([document(1), {"reject": True}, document(2)])
    assert (report["seen"], report["kept"], report["dropped"]) == (3, 2, 1)
    assert report["tokens_est"] == 200
    assert report["acceptance_rate"] == pytest.approx(2 / 3)
    assert report["buckets"]["general"] == {"tokens": 200, "cap": 10000, "progress": 0.02}
    assert report["languages"]["en"]["tokens"] == 200
    progress = capsys.readouterr().err
    for marker in ("general", "tokens", "seen=3", "kept=2", "dropped=1", "acceptance=", "tok/s=", "ETA="):
        assert marker in progress


def test_resume_preserves_shards_ids_text_and_corpus(setup, tmp_path):
    one, two, three = document(1), document(2), document(3)
    setup([one, two])
    output = tmp_path / "output"
    before = snapshot(output)
    corpus = tmp_path / "corpus/pilot_corpus.txt"
    prefix = corpus.read_bytes()
    # Same ID with changed text AND same text with a changed ID must be rejected.
    same_id = dict(document(99), id=one["id"])
    same_text = dict(two, id="new-id")
    report = setup([one, same_id, same_text, three], resume=True)
    assert report["resume"]["mode"] == "checkpoint"
    assert report["resume"]["initial_docs"] == 2
    assert report["kept"] == 3
    assert report["session"]["dropped"] == 3
    assert report["next_shard_idx"] == 2
    assert snapshot(output)["clean-00000.parquet"] == before["clean-00000.parquet"]
    assert corpus.read_bytes().startswith(prefix)
    assert corpus.read_text().splitlines() == [one["text"], two["text"], three["text"]]
    assert len({row["id"] for row in parquet_rows(output)}) == 3


@pytest.mark.parametrize("damage", ["missing", "stale", "broken", "index_missing"])
def test_checkpoint_fallback(setup, tmp_path, damage):
    setup([document(1), document(2)])
    state = tmp_path / "output/pilot_state.json"
    if damage == "missing":
        state.unlink()
    elif damage == "broken":
        state.write_text("{half-written")
    elif damage == "index_missing":
        (tmp_path / "output/pilot_index.sqlite3").unlink()
    else:
        publish_shard(tmp_path / "output", 9, [document(9, 403)])
    report = setup(resume=True)
    assert report["resume"]["mode"] == "parquet"
    rows = parquet_rows(tmp_path / "output")
    assert report["tokens_est"] == sum(estimate_tokens(row) for row in rows)
    assert report["kept"] == len(rows)
    assert report["next_shard_idx"] == (10 if damage == "stale" else 1)
    assert report["acceptance_rate"] is None
    assert report["seen_is_lower_bound"]


def test_checkpoint_fast_path_does_not_reconstruct(setup, monkeypatch):
    setup([document(1)])
    def no_reconstruct(_):
        pytest.fail("A valid checkpoint must not reread Parquet to reconstruct state")
    monkeypatch.setattr(PilotStore, "reconstruct", no_reconstruct)
    assert setup(resume=True)["resume"]["mode"] == "checkpoint"


def test_tail_closes_before_searching_for_smaller_doc(setup):
    def stream():
        yield document(1, 396)  # 99 tokens, one token remains.
        yield document(2, 400)
        pytest.fail("Builder kept searching after an oversized candidate")
    report = setup(stream(), max_tokens=100)
    assert report["tokens_est"] == 99
    assert report["source_stops"]["fineweb_edu_10bt"] == "bucket_document_boundary"
    assert report["stop_reason"] == "bucket_caps_or_tails"


def test_unconditional_small_tail_does_not_fetch_extra(setup):
    def stream():
        yield document(1, 3996)
        pytest.fail("Tail should close before fetching another raw row")
    report = setup(stream(), max_tokens=1000, tail_tokens=2048)
    assert report["tokens_est"] == 999
    assert report["source_stops"]["fineweb_edu_10bt"] == "bucket_tail"


def test_infinite_rejected_stream_is_bounded(setup):
    report = setup(itertools.repeat({"reject": True}), max_source_docs=7)
    assert report["seen"] == report["dropped"] == 7
    assert report["stop_reason"] == "source_scan_limit"


def test_language_cap_checks_whole_document(setup, monkeypatch):
    monkeypatch.setattr(builder, "PILOT_LANGUAGE_QUOTAS", {"en": 0.5, "ru": 0.5})
    report = setup([document(1, 196), document(2, 80)], max_tokens=100)
    assert report["languages"]["en"]["tokens"] == 49
    assert report["filter"]["dropped_lang_quota"] == 1
    assert report["tokens_est"] <= 100


def test_fresh_refuses_existing_and_atomic_publisher_cannot_clobber(setup, tmp_path):
    setup([document(1)])
    output = tmp_path / "output"
    before = snapshot(output)
    corpus = (tmp_path / "corpus/pilot_corpus.txt").read_bytes()
    with pytest.raises(FileExistsError):
        setup([document(2)])
    with pytest.raises(FileExistsError):
        publish_shard(output, 0, [document(2)])
    assert snapshot(output) == before
    assert (tmp_path / "corpus/pilot_corpus.txt").read_bytes() == corpus


def test_explicit_overwrite_and_invalid_flags(setup):
    setup([document(1)])
    report = setup([document(2)], overwrite=True)
    assert report["kept"] == 1
    with pytest.raises(ValueError, match="mutually exclusive"):
        setup(resume=True, overwrite=True)


def test_ctrl_c_after_completed_shard_and_partial_buffer(setup, tmp_path):
    def stream():
        yield document(1)
        yield document(2)
        yield document(3)
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        setup(stream())
    output = tmp_path / "output"
    before = snapshot(output)
    assert len(parquet_rows(output)) == 3
    assert json.loads((output / "pilot_report.json").read_text())["stop_reason"] == "interrupted"
    report = setup([document(i) for i in range(1, 5)], resume=True)
    assert report["kept"] == 4
    for name, digest in before.items():
        assert snapshot(output)[name] == digest


@pytest.mark.parametrize("crash_point", ["after_parquet", "after_corpus", "before_json"])
def test_crash_during_flush_reconstructs_without_duplicate_output(setup, tmp_path, monkeypatch, crash_point):
    import data.pilot_state as state_module
    setup([document(1), document(2)])
    if crash_point == "after_parquet":
        original = PilotStore.append_corpus
        def fail(*args, **kwargs):
            raise KeyboardInterrupt()
        monkeypatch.setattr(PilotStore, "append_corpus", fail)
    elif crash_point == "after_corpus":
        original = PilotStore.append_corpus
        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise KeyboardInterrupt()
        monkeypatch.setattr(PilotStore, "append_corpus", fail)
    else:
        original = state_module.atomic_json
        def fail(path, payload):
            if path.name == "pilot_state.json":
                raise KeyboardInterrupt()
            return original(path, payload)
        monkeypatch.setattr(state_module, "atomic_json", fail)
    with pytest.raises(KeyboardInterrupt):
        setup([document(3), document(4)], resume=True)
    if crash_point == "before_json":
        monkeypatch.setattr(state_module, "atomic_json", original)
    else:
        monkeypatch.setattr(PilotStore, "append_corpus", original)
    report = setup([document(i) for i in range(1, 5)], resume=True)
    assert report["kept"] == 4
    assert report["resume"]["mode"] == "parquet"
    assert len(parquet_rows(tmp_path / "output")) == 4
    lines = (tmp_path / "corpus/pilot_corpus.txt").read_text().splitlines()
    assert len(lines) == len(set(lines)) == 4


def test_partial_utf8_corpus_append_recovered_without_truncation(setup, tmp_path):
    setup([document(1)])
    output = tmp_path / "output"
    corpus = tmp_path / "corpus/pilot_corpus.txt"
    before = corpus.read_bytes()
    payload = "продолжение строки\n".encode()
    atomic_json(output / "pilot_corpus_journal.json", {
        "path": str(corpus.resolve()), "offset": len(before),
        "payload": base64.b64encode(payload).decode(),
    })
    with corpus.open("ab") as handle:
        handle.write(payload[:3])  # Kill halfway through a Cyrillic character.
    report = setup(resume=True)
    assert report["resume"]["mode"] == "parquet"
    assert corpus.read_bytes() == before + payload
    assert not (output / "pilot_corpus_journal.json").exists()


def test_resume_missing_corpus_recovered_from_shards(setup, tmp_path):
    setup([document(1), document(2)])
    corpus = tmp_path / "corpus/pilot_corpus.txt"
    corpus.unlink()
    setup(resume=True)
    assert corpus.read_text().splitlines() == [document(1)["text"], document(2)["text"]]


def test_output_lock_rejects_concurrent_builder(setup, tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    with build_lock(output / ".pilot.lock"):
        with pytest.raises(RuntimeError, match="Another pilot builder"):
            setup()


def test_ten_legacy_shards_40960_docs_and_exact_per_row_estimate(tmp_path):
    output, corpus_dir = tmp_path / "output", tmp_path / "corpus"
    output.mkdir()
    corpus_dir.mkdir()
    corpus = corpus_dir / "pilot_corpus.txt"
    prefix = b"previous tokenizer corpus must survive\n"
    corpus.write_bytes(prefix)
    n_docs, total_chars = 40_960, 193_712_334
    length, extra = divmod(total_chars, n_docs)
    expected = 0
    for shard in range(10):
        rows = [document(i, length + (i < extra)) for i in range(shard * 4096, (shard + 1) * 4096)]
        expected += sum(max(row["n_chars"] // 4, 1) for row in rows)
        pq.write_table(pa.Table.from_pylist(rows), output / f"clean-{shard:05d}.parquet")
    before = snapshot(output)
    store = PilotStore(output, corpus, corpus_limit=0)
    try:
        assert store.restore(True) == "parquet"
        state = store.state
        assert state["written_docs"] == 40960
        assert state["shard_idx"] == 10
        for field, key in (("bucket_counts", "general"), ("lang_counts", "en"),
                           ("source_counts", "fineweb_edu_10bt"), ("script_counts", "latin")):
            assert state[field][key] == expected
        assert state["total_tokens"] == expected
        assert 48_400_000 < expected <= 48_428_083
        assert store.restore(True) == "checkpoint"
        publish_shard(output, state["shard_idx"], [document("next")])
        assert (output / "clean-00010.parquet").exists()
        assert sum(pq.ParquetFile(p).metadata.num_rows for p in output.glob("clean-*.parquet")) == 40961
        assert corpus.read_bytes() == prefix
        for name, digest in before.items():
            assert snapshot(output)[name] == digest
    finally:
        store.close()


def test_default_rope_config_does_not_warn_or_mutate_input():
    spec = importlib.util.spec_from_file_location("standalone_shinra_config", "model/configuration_shinra.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = []
    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())
    logger = logging.getLogger("transformers")
    handler = Capture()
    logger.addHandler(handler)
    try:
        legacy = {"rope_type": "default", "factor": 1.0}
        for _ in range(10):
            assert "factor" not in module.ShinraConfig().rope_scaling
            assert "factor" not in module.ShinraConfig(rope_scaling=legacy).rope_scaling
        assert legacy["factor"] == 1.0
        assert module.ShinraConfig(rope_scaling={"rope_type": "linear", "factor": 2.0}).rope_scaling["factor"] == 2.0
    finally:
        logger.removeHandler(handler)
    assert not any("Unrecognized keys" in text for text in records)


def test_real_ingestion_has_no_transformers_or_shinra_imports(tmp_path):
    # Real filters and local source; importing model/transformers would fail this
    # subprocess, so a per-document config side effect cannot hide in mocks.
    code = r'''
import importlib.abc, os, random, sys, tempfile
from pathlib import Path
class NoModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'transformers', 'model'}:
            raise AssertionError('Unexpected model/config import: ' + fullname)
sys.meta_path.insert(0, NoModelImports())
os.environ.pop('SHINRA_TOXICITY_MODEL', None)
from data import clean
from data.build_pilot import build_pilot
vocabulary = 'people world system example morning evening family science engine garden school service community travel reading history network training language building market health window research useful develop motion source value meaning computer information provide human public design planet nature space water earth machine'.split()
rows=[]
for i in range(5):
    rng=random.Random(i)
    rows.append({'id': str(i), 'text': ' '.join(rng.choice(vocabulary) for _ in range(130))})
clean.iter_source=lambda *args: iter(rows)
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)
    report=build_pilot(root/'data', root/'corpus', 1000000, 1e9, 2, 100000, None)
    assert report['kept'] == 5, report
assert 'transformers' not in sys.modules
assert 'model' not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "rope_parameters" not in result.stderr


def test_persisted_minhash_still_rejects_near_duplicates(setup):
    original = document(1)
    original['text'] = ' '.join(f'word{n}' for n in range(100))
    original['n_chars'] = len(original['text'])
    changed = dict(original, id='different-id', text=original['text'].replace('word50', 'changedword'))
    changed['n_chars'] = len(changed['text'])
    setup([original])
    report = setup([changed], resume=True)
    assert report['kept'] == 1
    assert report['dedup']['dropped_near'] == 1


def test_document_boundary_stop_survives_normal_resume(setup):
    setup([document(1, 360), document(2, 400)], max_tokens=100)
    def stream():
        pytest.fail('Closed bucket should not reopen on a normal resume')
        yield
    report = setup(stream(), max_tokens=100, resume=True)
    assert report['tokens_est'] == 90
    assert report['session']['seen'] == 0
