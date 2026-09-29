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
    assert report["run_mode"] == "fixture"
    assert report["token_count_mode"] == "whitespace"


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
    assert report["over_budget"] == 0
    assert report["documents_kept"] == 0
    assert report["status"] == "fail"


def test_token_cap_stops_without_scanning_rest(tmp_path: Path):
    path = tmp_path / "many.jsonl"
    rec = {
        "text": "The clay path dried before noon and the archivist closed the grey ledger.",
        "source_id": "fixture-canary-en",
        "source_type": "natural",
        "domain": "general",
        "language": "en",
        "split": "train",
        "license": {"id": "fixture-local", "redistribution": True},
        "provenance": {
            "uri_hash": "sha256:" + "a" * 64,
            "snapshot": "phase-b-fixture",
        },
    }
    lines = []
    for i in range(40):
        item = dict(rec)
        item["text"] = (
            "one two three four five six seven"
            if i == 0
            else f"Unique canary row {i} copper filings stayed labelled drawer clerk drank water."
        )
        lines.append(json.dumps(item, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = run_canary(path, max_tokens=7)
    assert report["tokens_kept"] == 7
    assert report["over_budget"] == 0
    assert report["documents_kept"] == 1
    assert report["documents_seen"] == 1
    assert report["status"] == "pass"


def test_refuses_remote_source():
    with pytest.raises(PhaseBError, match="network"):
        ensure_local_path("https://example.com/corpus.jsonl")
    with pytest.raises(PhaseBError, match="network"):
        run_canary("hf://datasets/nope")


def test_engine_sources_have_no_network_clients():
    banned = (
        "huggingface_hub",
        "urllib.request",
        "requests.get",
        "import requests",
        "httpx",
        "datasets.load_dataset",
        "snapshot_download",
    )
    for path in PHASE_B_PY.glob("*.py"):
        if path.name in {"acquire_hf.py", "run_fineweb_edu_en.py"}:
            continue
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


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        if add_special_tokens:
            raise AssertionError("add_special_tokens must be False")
        ids: list[int] = []
        for _ in text.split():
            ids.extend([19, 20])
        return ids


def test_production_allowlist_is_closed():
    from data.data_v1.sources import assert_production_allowlist_governance, load_allowlist

    assert_production_allowlist_governance()
    doc = load_allowlist()
    assert [row["source_id"] for row in doc["sources"]] == ["fineweb-edu-en", "fineweb2-ru"]
    assert doc["downloaders"] is False
    assert doc["acquisition"] == "closed"
    assert doc["token_count"]["canary"] == "tokenizer_required"
    assert doc["token_count"]["add_special_tokens"] is False
    assert doc["disk"]["max_materialized_bytes"] == 8589934592
    for row in doc["sources"]:
        assert row["materialization"]["status"] == "not_materialized"
        assert row["materialization"]["content_sha256"] is None
        assert row["max_materialized_bytes"] == 4294967296
        assert row["expected_token_range"]["max"] == 8_000_000
        assert row["license"]["dataset_id"] == "ODC-By-1.0"
        assert row["license"]["dataset_redistribution"] is True
        assert row["license"]["underlying_content_caveat"] == "common_crawl_third_party_rights"
        assert row["license"]["common_crawl_tou"] is True
    assert doc["sources"][1]["upstream"]["subset"] == "rus_Cyrl"
    assert doc["sources"][0]["upstream"]["revision"] == "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    assert doc["sources"][1]["upstream"]["revision"] is None
    assert all(row["provenance_hash_strategy"] == "content_sha256" for row in doc["sources"])


def test_canary_mode_requires_tokenizer():
    with pytest.raises(PhaseBError, match="tokenizer"):
        run_canary(FIXTURES / "ok", mode="canary")


def test_canary_mode_fails_closed_allowlist():
    with pytest.raises(PhaseBError, match="materialization unresolved"):
        run_canary(FIXTURES / "ok", mode="canary", tokenizer=_FakeTokenizer())


def test_canary_mode_counts_tokenizer_tokens_on_stub_allowlist():
    report = run_canary(
        FIXTURES / "ok",
        mode="canary",
        tokenizer=_FakeTokenizer(),
        allowlist_path=FIXTURES / "allowlist_stub.json",
    )
    assert report["status"] == "pass"
    assert report["run_mode"] == "canary"
    assert report["token_count_mode"] == "tokenizer"
    assert report["tokens_kept"] == 116
    assert report["gates"]["token_count_mode"]["ok"] is True
    assert report["allowlist_misses"] == 0


def test_source_record_rejects_url_license():
    from data.data_v1.sources import load_allowlist, validate_source_record

    stub = load_allowlist(FIXTURES / "allowlist_stub.json")
    src = dict(stub["sources"][0])
    src["license"] = dict(src["license"])
    src["license"]["dataset_id"] = "https://example.com/license"
    with pytest.raises(PhaseBError, match="URL"):
        validate_source_record(src)


def test_unknown_source_id_rejected():
    from data.data_v1.sources import load_allowlist, validate_source_record

    src = dict(load_allowlist()["sources"][0])
    src["source_id"] = "wikipedia-en"
    with pytest.raises(PhaseBError, match="unknown source"):
        validate_source_record(src)


def test_fineweb2_wrong_subset_rejected():
    from data.data_v1.sources import load_allowlist, validate_source_record

    src = dict(load_allowlist()["sources"][1])
    src["upstream"] = dict(src["upstream"])
    src["upstream"]["subset"] = "eng_Latn"
    with pytest.raises(PhaseBError, match="wrong subset"):
        validate_source_record(src)


def test_source_language_mismatch_rejected():
    from data.data_v1.sources import load_allowlist, validate_source_record

    src = dict(load_allowlist()["sources"][1])
    src["languages"] = ["en"]
    with pytest.raises(PhaseBError, match="language mismatch"):
        validate_source_record(src)


def test_source_domain_outside_allowlist_rejected():
    from data.data_v1.sources import load_allowlist, validate_source_record

    src = dict(load_allowlist()["sources"][0])
    src["allowed_domains"] = ["general", "knowledge", "longform", "code"]
    with pytest.raises(PhaseBError, match="domain outside allowlist"):
        validate_source_record(src)


def test_source_over_4_gib_rejected():
    from data.data_v1.sources import load_allowlist, validate_source_record

    src = dict(load_allowlist()["sources"][0])
    src["max_materialized_bytes"] = 4294967296 + 1
    with pytest.raises(PhaseBError, match="4 GiB"):
        validate_source_record(src)


def test_aggregate_over_8_gib_rejected():
    from data.data_v1.sources import load_allowlist, validate_allowlist

    doc = json.loads(json.dumps(load_allowlist()))
    doc["disk"]["max_materialized_bytes"] = 8589934592 - 1
    with pytest.raises(PhaseBError, match="aggregate materialization"):
        validate_allowlist(doc)


def test_unresolved_hash_cannot_enter_real_canary():
    from data.data_v1.sources import assert_canary_materialized, load_allowlist

    with pytest.raises(PhaseBError, match="materialization unresolved"):
        assert_canary_materialized(load_allowlist())


def test_empty_local_input_rejected():
    with pytest.raises(PhaseBError, match="source not found"):
        run_canary(FIXTURES / "does-not-exist.jsonl")


def test_phase_a_hashes_unchanged():
    from data.data_v1.phase_a import validate_phase_a

    report = validate_phase_a()
    assert report["manifest_sha256"] == "c18fba5f054475fc9f87242aa048039e1ae9e3cbc860852af152647ef7d035dc"
    assert report["probe_bundle_sha256"] == "eb6d0b7552f646e2782d744ae0542f240fb0c350f980b05fb6a337b7bffc87b9"


def test_assert_allowed_sidecar_unknown_and_mismatch():
    from data.data_v1.phase_b import assert_allowed_sidecar
    from data.data_v1.sources import load_allowlist, source_index

    idx = source_index(load_allowlist())
    with pytest.raises(PhaseBError, match="unknown source"):
        assert_allowed_sidecar(
            {"source_id": "nope", "language": "en", "domain": "general", "source_type": "natural"},
            idx,
        )
    with pytest.raises(PhaseBError, match="language mismatch"):
        assert_allowed_sidecar(
            {"source_id": "fineweb-edu-en", "language": "ru", "domain": "general", "source_type": "natural"},
            idx,
        )
    with pytest.raises(PhaseBError, match="domain outside allowlist"):
        assert_allowed_sidecar(
            {"source_id": "fineweb-edu-en", "language": "en", "domain": "code", "source_type": "natural"},
            idx,
        )


def test_source_over_4_gib_rejected_for_both_ids():
    from data.data_v1.sources import load_allowlist, validate_source_record

    doc = load_allowlist()
    for src in doc["sources"]:
        row = dict(src)
        row["max_materialized_bytes"] = 4294967296 + 1
        with pytest.raises(PhaseBError, match="4 GiB"):
            validate_source_record(row)


def test_fineweb_edu_invented_subset_rejected():
    from data.data_v1.sources import load_allowlist, validate_source_record

    src = dict(load_allowlist()["sources"][0])
    src["upstream"] = dict(src["upstream"])
    src["upstream"]["subset"] = "sample-10BT"
    with pytest.raises(PhaseBError, match="subset"):
        validate_source_record(src)


def test_unverified_revision_rejected():
    from data.data_v1.sources import load_allowlist, validate_source_record

    src = dict(load_allowlist()["sources"][0])
    src["upstream"] = dict(src["upstream"])
    src["upstream"]["revision"] = "deadbeef"
    with pytest.raises(PhaseBError, match="unverified"):
        validate_source_record(src)


def test_allowlist_has_no_fake_snapshot_string():
    from data.data_v1.sources import ALLOWLIST_PATH, SCHEMA_PATH

    assert "PENDING-LOCAL-SLICE" not in ALLOWLIST_PATH.read_text(encoding="utf-8")
    assert "PENDING-LOCAL-SLICE" not in SCHEMA_PATH.read_text(encoding="utf-8")


def test_empty_directory_rejected(tmp_path):
    empty = tmp_path / "empty-dir"
    empty.mkdir()
    with pytest.raises(PhaseBError, match="no local jsonl"):
        run_canary(empty)


def test_empty_jsonl_fails_kept_gate(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    report = run_canary(path)
    assert report["status"] == "fail"
    assert report["documents_kept"] == 0
    assert report["gates"]["kept_nonempty"]["ok"] is False


def test_canary_accepts_scratch_receipt(tmp_path):
    from data.data_v1.materialize import materialize
    from data.data_v1.sources import load_allowlist

    jsonl = tmp_path / "in.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "text": "The clay path dried before noon and the archivist closed the grey ledger.",
                "source_id": "fineweb-edu-en",
                "source_type": "natural",
                "domain": "general",
                "language": "en",
                "split": "train",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    allow = load_allowlist()
    allow["disk"] = dict(allow["disk"], min_free_gb=0)
    allow_path = tmp_path / "allow.json"
    allow_path.write_text(json.dumps(allow), encoding="utf-8")
    state = materialize(
        jsonl,
        "fineweb-edu-en",
        scratch_root=tmp_path / "data_v1",
        allowlist=allow,
        free_bytes=50 * 1024**3,
    )
    report = run_canary(state["path"], mode="canary", tokenizer=_FakeTokenizer(), allowlist_path=allow_path)
    assert report["status"] == "pass"
    assert report["run_mode"] == "canary"
    assert report["token_count_mode"] == "tokenizer"
