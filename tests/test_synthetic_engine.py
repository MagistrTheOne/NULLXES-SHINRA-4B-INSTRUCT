from __future__ import annotations

from configs.loader import load_config
from data.synthetic_engine import _corpus_config_sha256, generate_corpus


def test_corpus_config_identity_is_stable_when_artifact_sha_pins_are_frozen():
    config = load_config()
    initial = _corpus_config_sha256(config)

    config["data"]["manifest_sha256"] = "a" * 64
    config["tokenizer"]["native_model_sha256"] = "b" * 64
    config["tokenizer"]["artifact_sha256"] = "c" * 64
    config["data"]["synthetic"]["status"] = "runtime_verified"
    assert _corpus_config_sha256(config) == initial

    config["data"]["synthetic"]["root_seed"] += 1
    assert _corpus_config_sha256(config) != initial


def test_small_corpus_generation_is_reproducible_and_reports_exact_repeats(tmp_path):
    first = generate_corpus(tmp_path / "first", records_per_family=2, root_seed=73)
    second = generate_corpus(tmp_path / "second", records_per_family=2, root_seed=73)

    assert first == second
    assert first["record_counts"] == {"train": 8, "validation": 8, "blind": 8}
    assert first["exact_duplicate_text_records"]["train"] >= 2
    assert first["unique_canonical_prompt_signatures"]["train"] < first["record_counts"]["train"]
    assert first["split_integrity"]["family_isolation"] is True
    assert first["split_integrity"]["template_isolation"] is True
    assert first["split_integrity"]["scenario_isolation"] is True
