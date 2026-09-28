"""Phase A: sidecar schema, frozen probes, contamination, S0 baseline. CPU only."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from data.data_v1.phase_a import (
    BASELINE_DIR,
    DATA_SPEC_PATH,
    DOMAINS,
    GENERATION_CONTRACT,
    PROBE_DIR,
    PROBE_FILES,
    PROBE_SPEC_PATH,
    SIDECAR_SCHEMA_PATH,
    PhaseAError,
    assert_text_not_contaminated,
    document_id_for_text,
    fingerprint_set,
    load_all_probes,
    score_generation,
    validate_phase_a,
    validate_probe_record,
    validate_sidecar,
    validate_sidecar_schema_file,
)

ROOT = Path(__file__).resolve().parents[1]


def _sidecar() -> dict:
    body = "The clay path dried before noon."
    doc_id = document_id_for_text(body)
    return {
        "schema_version": "shinra-data-v1",
        "document_id": doc_id,
        "source_id": "example-general-en",
        "source_type": "natural",
        "domain": "general",
        "language": "en",
        "split": "train",
        "license": {"id": "CC-BY-4.0", "redistribution": True},
        "provenance": {
            "uri_hash": "sha256:" + "ab" * 32,
            "snapshot": "2026-09-28",
        },
        "quality": {"score": 0.91, "flags": []},
        "dedup": {"exact_hash": doc_id, "near_group": None},
        "stats": {"chars": len(body), "tokens": 7},
    }


def test_sidecar_schema_file_is_closed():
    validate_sidecar_schema_file()
    schema = json.loads(SIDECAR_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["properties"]["split"]["enum"] == ["train", "validation"]
    assert "probe" not in schema["properties"]["split"]["enum"]
    assert schema["properties"]["domain"]["enum"] == list(DOMAINS)


def test_valid_sidecar_passes():
    validate_sidecar(_sidecar())


def test_sidecar_rejects_open_domain():
    doc = _sidecar()
    doc["domain"] = "interesting_science_blog"
    with pytest.raises(PhaseAError, match="domain"):
        validate_sidecar(doc)


def test_sidecar_rejects_probe_split():
    doc = _sidecar()
    doc["split"] = "probe"
    with pytest.raises(PhaseAError, match="train\\|validation"):
        validate_sidecar(doc)


def test_sidecar_rejects_raw_url():
    doc = _sidecar()
    doc["provenance"]["snapshot"] = "https://example.com/dump"
    with pytest.raises(PhaseAError, match="URL"):
        validate_sidecar(doc)


def test_sidecar_rejects_s0_v0_generator():
    doc = _sidecar()
    doc["source_type"] = "synthetic"
    doc["generator"] = "language_core.v0"
    with pytest.raises(PhaseAError, match="v0"):
        validate_sidecar(doc)


def test_document_id_is_content_addressed():
    a = document_id_for_text("Hello\nworld")
    b = document_id_for_text("Hello\nworld")
    c = document_id_for_text("Hello world")
    assert a.startswith("sha256:")
    assert a == b
    assert a != c


def test_all_probe_files_exist_and_validate():
    rows = load_all_probes()
    assert len(rows) >= 60
    families = {row["family"] for row in rows}
    assert families == {
        "completion_en",
        "completion_ru",
        "syntax",
        "negation",
        "coreference",
        "semantic",
        "math",
        "code",
        "structured",
        "long_context",
        "ood",
        "negative_identity",
        "negative_chat",
    }
    for name in PROBE_FILES:
        assert (PROBE_DIR / name).is_file()


def test_probe_record_requires_expected_for_numeric():
    row = {
        "id": "math_bad",
        "family": "math",
        "language": "en",
        "prompt": "1+1 is",
        "scorer": "numeric_exact",
        "max_new_tokens": 8,
        "tags": ["math"],
    }
    with pytest.raises(PhaseAError, match="expected"):
        validate_probe_record(row, "math.jsonl")


def test_contamination_firewall_blocks_probe_prompt_and_concat():
    rows = load_all_probes()
    fps = fingerprint_set(rows)
    math = next(r for r in rows if r["id"] == "math_00001")
    with pytest.raises(PhaseAError, match="fingerprint"):
        assert_text_not_contaminated(math["prompt"], fps)
    with pytest.raises(PhaseAError, match="fingerprint"):
        assert_text_not_contaminated(f"{math['prompt']} {math['expected']}", fps)
    assert_text_not_contaminated("Ordinary clay dried on the path before noon.", fps)


def test_scorers_match_s0_session():
    rows = {r["id"]: r for r in load_all_probes()}
    math = score_generation(rows["math_00002"], "11.<|end_of_text|>")
    assert math["pass"] is False
    assert "11" in math["detail"]
    ood = score_generation(rows["ood_00001"], "Alina counted them twice.<|end_of_text|>")
    assert ood["pass"] is False
    ident = score_generation(rows["nid_00001"], "<|end_of_text|>")
    assert ident["pass"] is True
    chat = score_generation(rows["nchat_00001"], '11, "ok": true}<|end_of_text|>')
    assert chat["pass"] is True
    logged = score_generation(rows["ood_00005"], "near the door...<|end_of_text|>")
    assert logged["pass"] is None
    choice = score_generation(rows["syn_00001"], "B")
    assert choice["pass"] is True


def test_generation_contract_forbids_cache_and_sampling():
    assert GENERATION_CONTRACT == {"do_sample": False, "use_cache": False, "max_new_tokens": 128}


def test_probe_and_data_specs_exist():
    assert DATA_SPEC_PATH.is_file()
    assert PROBE_SPEC_PATH.is_file()
    text = PROBE_SPEC_PATH.read_text(encoding="utf-8").casefold()
    assert "use_cache" in text
    assert "false" in text


def test_s1_language_yaml_untouched_by_phase_a():
    path = ROOT / "configs" / "stages" / "s1_language.yaml"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "s1" in text.lower()


def test_phase_a_static_audit():
    report = validate_phase_a()
    assert report["ok"] is True
    assert report["n_probes"] >= 60
    assert report["baseline_coverage"] == "partial"
    assert (BASELINE_DIR / "generations.jsonl").is_file()
    assert (BASELINE_DIR / "scores.json").is_file()
    assert (BASELINE_DIR / "run_manifest.json").is_file()


def test_probe_v1_is_immutable_via_manifest_hash():
    body = json.loads((PROBE_DIR / "manifest.json").read_text(encoding="utf-8"))
    mutated = copy.deepcopy(body)
    mutated["generation"]["use_cache"] = True
    assert mutated != body
