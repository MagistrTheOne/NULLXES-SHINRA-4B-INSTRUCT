"""Lineage invariant, S0–S2 data DNA, eos is not document stop."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from data.shard_lifecycle import build_packed_shard
from data.pack import PackContractError, wrap_pretrain_document

ROOT = Path(__file__).resolve().parents[1]

INVARIANT = (
    "SHINRA is SHINRA. Created by NULLXES. Compatibility with other model families "
    "is not SHINRA's identity or lineage. Hugging Face compatibility is an interface "
    "property, not model ancestry."
)

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "outputs",
    "tests",
    ".pytest_cache",
    ".cursor",
}

S0_S2_FILES = [
    ROOT / "data" / "rolling_drum.py",
    ROOT / "data" / "pack.py",
    ROOT / "data" / "shard_lifecycle.py",
    ROOT / "data" / "shards.py",
    ROOT / "data" / "ledger.py",
    ROOT / "scripts" / "v2_s0_colab.py",
    ROOT / "scripts" / "v2_stage_run.py",
    ROOT / "scripts" / "fetch_tokenizer.py",
    ROOT / "evaluation" / "v2_gates.py",
    ROOT / "training" / "trainer.py",
    ROOT / "training" / "collator.py",
    ROOT / "training" / "arguments.py",
    ROOT / "notebooks" / "SHINRA_V2_S0.ipynb",
]

S0_S2_DIRS = [
    ROOT / "data" / "synth",
    ROOT / "configs" / "stages",
]

S0_S2_FORBIDDEN = (
    "apply_chat_template",
    "identity_record",
    "from_pretrained(\"Qwen",
    "from_pretrained('Qwen",
)


def _iter_doc_files():
    suffixes = {".md", ".rst", ".ipynb", ".yaml", ".yml"}
    extra = {ROOT / "model" / "configuration_shinra.py"}
    for path in extra:
        if path.is_file():
            yield path
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES or part.startswith("pytest-cache") for part in path.parts):
            continue
        if path.suffix.lower() in suffixes:
            yield path


def test_invariant_sentence_frozen():
    text = (ROOT / "docs" / "SHINRA_INVARIANT.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    design = (ROOT / "architecture" / "design.md").read_text(encoding="utf-8")
    assert INVARIANT in text
    assert INVARIANT in readme
    assert INVARIANT in design


def test_docs_do_not_name_foreign_q_family():
    needle = "qwen"
    hits = []
    for path in _iter_doc_files():
        blob = path.read_text(encoding="utf-8", errors="ignore").lower()
        if needle in blob:
            hits.append(str(path.relative_to(ROOT)))
    assert hits == [], f"documentation still names a foreign family: {hits}"


def test_readme_is_v2_s0_s2_drum():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "v2_s0_colab.py" in readme
    assert "SHINRA_V2_S0.ipynb" in readme
    assert "pip install -e ." in readme
    assert 'pip install -e ".[train,flash]"' not in readme
    assert "prepare_data.sh" not in readme
    assert "SHINRA_COLAB.ipynb" not in readme


def test_s0_s2_sources_stay_pretrain_dna():
    files = list(S0_S2_FILES)
    for folder in S0_S2_DIRS:
        files.extend(p for p in folder.rglob("*") if p.is_file() and p.suffix in {".py", ".yaml", ".yml", ".md", ".ipynb"})
    hits = []
    for path in files:
        blob = path.read_text(encoding="utf-8", errors="ignore")
        for needle in S0_S2_FORBIDDEN:
            if needle in blob:
                hits.append(f"{path.relative_to(ROOT)}:{needle}")
    assert hits == [], f"S0–S2 path contamination: {hits}"


def test_yaml_keeps_eos_and_document_end_split():
    model = yaml.safe_load((ROOT / "configs" / "shinra_4b.yaml").read_text(encoding="utf-8"))["model"]
    assert model["eos_token_id"] == 2
    assert model["document_end_token_id"] == 18
    assert model["eos_token_id"] != model["document_end_token_id"]


def test_pipeline_does_not_pass_eos_as_end_id():
    blobs = []
    for rel in (
        "data/pack.py",
        "data/shard_lifecycle.py",
        "scripts/v2_stage_run.py",
        "training/trainer.py",
        "training/collator.py",
    ):
        blobs.append((rel, (ROOT / rel).read_text(encoding="utf-8")))
    for rel, blob in blobs:
        assert "end_id=config.eos_token_id" not in blob
        assert "end_id=cfg.eos_token_id" not in blob
        assert "end_id = config.eos_token_id" not in blob
        assert "end_id = cfg.eos_token_id" not in blob


def test_build_packed_shard_rejects_eot_end(tmp_path: Path):
    with pytest.raises(ValueError, match="END_OF_TEXT"):
        build_packed_shard(
            stage="s0",
            shard_index=0,
            seed=0,
            record_count=4,
            encode=lambda text: [19, 20],
            output_dir=tmp_path,
            sequence_length=16,
            end_id=2,
        )


def test_wrap_default_terminator_is_end_of_text_not_eot():
    wrapped = wrap_pretrain_document([19, 20])
    assert wrapped[-1] == 18
    with pytest.raises(PackContractError):
        wrap_pretrain_document([19, 20], end_id=2)
