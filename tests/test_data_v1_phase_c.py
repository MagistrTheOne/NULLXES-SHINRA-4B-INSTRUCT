"""Phase C packer CPU contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from data.data_v1.phase_c import (
    CEILING_HONEST_TOKENS,
    HUB_STATUS,
    RESUME_RELATIVE,
    RESUME_STEP,
    S1_STATUS,
    SOURCE_ID,
    TARGET_HONEST_TOKENS,
    TRAIN_AUTHORIZED,
    PhaseCError,
    main,
    pack_corpus,
    run_train,
)
from tokenizer.special_tokens import PRETRAIN_FORBIDDEN_IDS

ROOT = Path(__file__).resolve().parents[1]
PHASE_C = ROOT / "data" / "data_v1" / "phase_c.py"
S1_YAML = ROOT / "configs" / "stages" / "s1_language.yaml"


def _encode(text: str) -> list[int]:
    words = text.split()
    if not words:
        return []
    return [19 + (i % 40) for i, _ in enumerate(words)]


def _jsonl(path: Path, texts: list[str]) -> Path:
    lines = []
    for i, text in enumerate(texts):
        lines.append(
            json.dumps(
                {
                    "text": text,
                    "source_id": "fineweb-edu-en",
                    "n": i,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _words(n: int, tag: str) -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def test_frozen_pilot_numbers():
    assert TARGET_HONEST_TOKENS == 8_000_000
    assert CEILING_HONEST_TOKENS == 10_000_000
    assert SOURCE_ID == "fineweb-edu-en"
    assert RESUME_RELATIVE == "s0/final/step-00001358"
    assert RESUME_STEP == 1358
    assert TRAIN_AUTHORIZED is True
    assert S1_STATUS == "closed"
    assert HUB_STATUS == "closed"


def test_train_cli_is_colab_runner_not_this_module():
    with pytest.raises(PhaseCError, match="v2_c_colab"):
        run_train()
    assert main(["train"]) == 2
    assert (ROOT / "scripts" / "v2_c_colab.py").is_file()


def test_s1_yaml_untouched_and_not_referenced():
    text = PHASE_C.read_text(encoding="utf-8")
    assert "s1_language" not in text
    assert S1_YAML.is_file()


def test_pack_hits_target_and_writes_bin(tmp_path: Path):
    jsonl = _jsonl(tmp_path / "docs.jsonl", [_words(6, f"row{i}-") for i in range(12)])
    out = tmp_path / "packed"
    report = pack_corpus(
        input_path=jsonl,
        output_dir=out,
        encode=_encode,
        sequence_length=8,
        target_honest_tokens=24,
        ceiling_honest_tokens=40,
    )
    assert report["status"] == "packed"
    assert report["honest_tokens"] >= 24
    assert report["honest_tokens"] <= 40
    assert report["resume_from"] == RESUME_RELATIVE
    bin_path = out / "shard-00000.bin"
    assert bin_path.is_file()
    meta = json.loads(bin_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["packer"] == "shinra-v2-pack.v1"
    ids = report["shard"]
    assert ids is not None
    receipt = json.loads((out / "phase_c.pack.json").read_text(encoding="utf-8"))
    assert receipt["train_authorized"] is True
    assert receipt["s1"] == "closed"


def test_pack_is_deterministic(tmp_path: Path):
    jsonl = _jsonl(tmp_path / "docs.jsonl", [_words(8, f"det{i}-") for i in range(6)])
    a = tmp_path / "a"
    b = tmp_path / "b"
    kwargs = dict(
        input_path=jsonl,
        encode=_encode,
        sequence_length=8,
        target_honest_tokens=16,
        ceiling_honest_tokens=32,
    )
    ra = pack_corpus(output_dir=a, **kwargs)
    rb = pack_corpus(output_dir=b, **kwargs)
    assert ra["shard"]["sha256"] == rb["shard"]["sha256"]
    assert ra["honest_tokens"] == rb["honest_tokens"]


def test_wrap_rejects_eot_and_chat_ids_by_skipping(tmp_path: Path):
    def encode(text: str) -> list[int]:
        if "poison" in text:
            return [19, 2, 20]
        return [19, 20, 21, 22, 23, 24]

    jsonl = _jsonl(
        tmp_path / "docs.jsonl",
        [
            "poison document should skip",
            _words(6, "clean-"),
            _words(6, "also-"),
            _words(6, "more-"),
            _words(6, "last-"),
        ],
    )
    report = pack_corpus(
        input_path=jsonl,
        output_dir=tmp_path / "packed",
        encode=encode,
        sequence_length=8,
        target_honest_tokens=16,
        ceiling_honest_tokens=32,
    )
    assert report["documents_skipped"] >= 1
    assert report["honest_tokens"] >= 16
    packed = json.loads((tmp_path / "packed" / "shard-00000.meta.json").read_text(encoding="utf-8"))
    raw = (tmp_path / "packed" / "shard-00000.bin").read_bytes()
    arr = np.frombuffer(raw, dtype=np.uint32).reshape(packed["shape"])
    flat = set(int(x) for x in arr.reshape(-1))
    assert 2 not in flat
    assert not (PRETRAIN_FORBIDDEN_IDS & flat)


def test_ceiling_blocks_overshoot(tmp_path: Path):
    jsonl = _jsonl(tmp_path / "docs.jsonl", [_words(6, f"cap{i}-") for i in range(20)])
    report = pack_corpus(
        input_path=jsonl,
        output_dir=tmp_path / "packed",
        encode=_encode,
        sequence_length=8,
        target_honest_tokens=16,
        ceiling_honest_tokens=16,
    )
    assert report["status"] == "packed"
    assert report["honest_tokens"] == 16
    assert report["honest_tokens"] <= report["ceiling_honest_tokens"]


def test_under_target_fails(tmp_path: Path):
    jsonl = _jsonl(tmp_path / "docs.jsonl", [_words(6, "only-")])
    with pytest.raises(PhaseCError, match="pack failed"):
        pack_corpus(
            input_path=jsonl,
            output_dir=tmp_path / "packed",
            encode=_encode,
            sequence_length=8,
            target_honest_tokens=24,
            ceiling_honest_tokens=40,
        )


def test_refuses_remote_source(tmp_path: Path):
    with pytest.raises(PhaseCError):
        pack_corpus(
            input_path="https://example.com/x.jsonl",
            output_dir=tmp_path,
            encode=_encode,
            sequence_length=8,
            target_honest_tokens=8,
            ceiling_honest_tokens=16,
        )


def test_engine_has_no_network_clients():
    text = PHASE_C.read_text(encoding="utf-8")
    for needle in (
        "huggingface_hub",
        "urllib.request",
        "requests.get",
        "import requests",
        "httpx",
        "datasets.load_dataset",
        "snapshot_download",
        "from_pretrained",
        "s1_language.yaml",
    ):
        assert needle not in text


def test_bos_and_end_of_text_in_stream(tmp_path: Path):
    jsonl = _jsonl(tmp_path / "docs.jsonl", [_words(5, "wrap-") for _ in range(8)])
    report = pack_corpus(
        input_path=jsonl,
        output_dir=tmp_path / "packed",
        encode=_encode,
        sequence_length=8,
        target_honest_tokens=16,
        ceiling_honest_tokens=32,
    )
    meta = json.loads((tmp_path / "packed" / "shard-00000.meta.json").read_text(encoding="utf-8"))
    arr = np.frombuffer((tmp_path / "packed" / "shard-00000.bin").read_bytes(), dtype=np.uint32)
    arr = arr.reshape(meta["shape"])
    flat = [int(x) for x in arr.reshape(-1)]
    assert 1 in flat
    assert 18 in flat
    assert report["honest_tokens"] == int((arr != 3).sum())
