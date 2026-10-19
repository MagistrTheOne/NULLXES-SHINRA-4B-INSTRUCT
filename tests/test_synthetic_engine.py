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
    assert first["record_counts"] == {"train": 16, "validation": 12, "blind": 12}
    assert first["exact_duplicate_text_records"]["train"] == 0
    assert first["unique_canonical_prompt_signatures"]["train"] < first["record_counts"]["train"]
    assert first["record_counts_by_language"] == {
        "train": {"en": 8, "ru": 8},
        "validation": {"en": 6, "ru": 6},
        "blind": {"en": 6, "ru": 6},
    }
    assert first["text_length_characters"]["train"]["count"] == 16
    assert first["split_integrity"]["family_isolation"] is True
    assert first["split_integrity"]["template_isolation"] is True
    assert first["split_integrity"]["scenario_isolation"] is True
