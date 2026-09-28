"""Phase B corpus canary: local ingest, hard gates, no network. CPU only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.data_v1 import PhaseBError
from data.data_v1.dedup import CanaryDedup
from data.data_v1.ingest import ensure_local_path, iter_records
from data.data_v1.phase_b import run_canary
from data.data_v1.report import MAX_CANARY_TOKENS, REPORT_KEYS

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "data_v1" / "phase_b"
PHASE_B_PY = ROOT / "data" / "data_v1"
S1_YAML = ROOT / "configs" / "stages" / "s1_language.yaml"


def test_clean_local_canary_passes():
    report = run_canary(FIXTURES / "ok")
    assert report["status"] == "pass"
    assert report["documents_kept"] >= 3
    assert report["tokens_kept"] > 0
    assert report["tokens_kept"] <= MAX_CANARY_TOKENS
    assert report["probe_collisions"] == 0
    assert report["special_token_violations"] == 0
    assert report["s0_v0_generators"] == 0
    assert report["raw_url_sidecar"] == 0
    assert report["languages"]["en"] >= 1
    assert report["languages"]["ru"] >= 1
    assert set(report["languages"]) <= {"en", "ru"}
    assert set(report["domains"]) <= {
        "general",
        "longform",
        "knowledge",
        "semantic",
        "code",
        "math",
        "reasoning",
        "structured",
    }
    for key in REPORT_KEYS:
        assert key in report
    assert report["training"] is False
    assert report["network"] is False


def test_probe_overlap_fails_gate():
    report = run_canary(FIXTURES / "bad" / "probe.jsonl")
    assert report["status"] == "fail"
    assert report["probe_collisions"] >= 1
    assert report["documents_kept"] == 0
    assert report["gates"]["probe_overlap"]["ok"] is False


def test_s0_v0_generator_forbidden():
    report = run_canary(FIXTURES / "bad" / "v0.jsonl")
    assert report["status"] == "fail"
    assert report["s0_v0_generators"] >= 1
    assert report["gates"]["s0_v0_generators"]["ok"] is False


def test_raw_url_in_sidecar_forbidden():
    report = run_canary(FIXTURES / "bad" / "url.jsonl")
    assert report["status"] == "fail"
    assert report["raw_url_sidecar"] >= 1


def test_special_tokens_in_body_forbidden():
    report = run_canary(FIXTURES / "bad" / "specials.jsonl")
    assert report["status"] == "fail"
    assert report["special_token_violations"] >= 1


def test_open_domain_forbidden():
    report = run_canary(FIXTURES / "bad" / "domain.jsonl")
    assert report["status"] == "fail"
    assert report["domain_violations"] >= 1
    assert report["gates"]["domain"]["ok"] is False


def test_probe_is_not_a_split():
    report = run_canary(FIXTURES / "bad" / "split.jsonl")
    assert report["status"] == "fail"
    assert report["split_violations"] >= 1
    assert report["gates"]["split"]["ok"] is False


def test_language_must_be_en_or_ru():
    report = run_canary(FIXTURES / "bad" / "language.jsonl")
    assert report["status"] == "fail"
    assert report["language_violations"] >= 1
    assert report["gates"]["language"]["ok"] is False


def test_exact_duplicate_is_counted_not_kept_twice():
    report = run_canary(FIXTURES / "dups.jsonl")
    assert report["documents_seen"] == 2
    assert report["documents_kept"] == 1
    assert report["exact_duplicates"] >= 1
    assert report["status"] == "pass"


def test_near_duplicate_drops_second_doc():
    report = run_canary(FIXTURES / "near.jsonl")
    assert report["documents_seen"] == 2
    assert report["documents_kept"] == 1
    assert report["near_duplicates"] >= 1 or report["exact_duplicates"] >= 1


def test_token_cap_cannot_be_raised():
    with pytest.raises(PhaseBError, match="cannot raise"):
        run_canary(FIXTURES / "ok", max_tokens=MAX_CANARY_TOKENS + 1)


def test_token_cap_hard_gate():
    report = run_canary(FIXTURES / "ok" / "ok.jsonl", max_tokens=5)
    assert report["tokens_kept"] <= 5
    assert report["status"] == "fail"
    assert report["over_budget"] >= 1 or report["documents_kept"] == 0


def test_refuses_remote_source():
    with pytest.raises(PhaseBError, match="network"):
        ensure_local_path("https://example.com/corpus.jsonl")
    with pytest.raises(PhaseBError, match="network"):
        run_canary("hf://datasets/nope")


def test_engine_sources_have_no_network_clients():
    banned = ("huggingface_hub", "urllib.request", "requests.get", "httpx", "datasets.load_dataset")
    for path in PHASE_B_PY.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in banned:
            assert needle not in text, f"{path.name} contains {needle}"


def test_s1_language_yaml_still_unauthorized_legacy():
    text = S1_YAML.read_text(encoding="utf-8")
    assert "230" in text
    assert "synth" in text.lower() or "s1" in text.lower()


def test_iter_records_reads_jsonl_and_txt():
    rows = list(iter_records(FIXTURES / "ok"))
    assert len(rows) >= 4
    assert all("text" in row for row in rows)


def test_canary_dedup_exact_unit():
    idx = CanaryDedup()
    text = "The clay path dried before noon and the archivist closed the grey ledger."
    assert idx.check(text)[0] == "keep"
    assert idx.check(text)[0] == "exact"
