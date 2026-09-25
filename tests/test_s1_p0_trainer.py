"""Static P0 trainer checks. Tiny tensors only. No model weights."""

from __future__ import annotations

import json
from pathlib import Path

import importlib.util

import torch


def _load(module_name: str):
    path = Path(__file__).resolve().parents[1] / "training" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checkpoint = _load("s1_p0_checkpoint")
loss = _load("s1_p0_loss")
reader = _load("s1_p0_reader")
tokens = _load("s1_p0_tokens")
CheckpointError = checkpoint.CheckpointError
load_resume = checkpoint.load_resume
publish_latest = checkpoint.publish_latest
write_generation = checkpoint.write_generation
ObjectiveError = loss.ObjectiveError
scale_gradients = loss.scale_gradients
supervised_sums = loss.supervised_sums
DatasetPinError = reader.DatasetPinError
take_update = reader.take_update
require_pins = reader.require_pins
verify_authority = reader.verify_authority
verify_shard = reader.verify_shard
TokenizerContractError = tokens.TokenizerContractError
labels_from_ids = tokens.labels_from_ids


def test_prompt_mask_and_eot_supervision():
    prompt = [1, 10, 11]
    full = [1, 10, 11, 20, 2]
    row = labels_from_ids(prompt, full)
    assert row["labels"][:3] == [-100, -100, -100]
    assert row["labels"][3:] == [20, 2]
    assert row["target_tokens"] == 2


def test_shift_count_matches_target_tokens():
    labels = torch.tensor([[-100, -100, 7, 2]])
    logits = torch.zeros(1, 4, 8)
    _, _, count = supervised_sums(logits, labels)
    assert count == 2


def test_denominator_scales_once():
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad = torch.tensor([4.0, 8.0])
    scale_gradients([parameter], 4)
    assert torch.equal(parameter.grad, torch.tensor([1.0, 2.0]))


def test_overlength_rejected():
    try:
        labels_from_ids([1, 4], [1, 4] + [5] * 8191)
    except TokenizerContractError:
        return
    raise AssertionError("overlength accepted")


def test_double_bos_rejected():
    try:
        labels_from_ids([1, 1, 4], [1, 1, 4, 2])
    except TokenizerContractError:
        return
    raise AssertionError("double BOS accepted")


def test_nan_loss_rejected():
    logits = torch.tensor([[[float("nan"), 0.0], [0.0, 0.0]]])
    labels = torch.tensor([[-100, 0]])
    try:
        supervised_sums(logits, labels)
    except ObjectiveError:
        return
    raise AssertionError("nan loss accepted")


def test_update_plan_replay_and_order(tmp_path: Path):
    row = {"target_tokens": 20, "id": "a"}
    line = json.dumps(row) + "\n"
    shard = tmp_path / "train-00000.jsonl"
    shard.write_text(line * 2, encoding="utf-8")
    blob = shard.read_bytes()
    import hashlib

    digest = hashlib.sha256(blob).hexdigest()
    manifest = {
        "dataset_repo": "MagistrTheOne/NULLXES-SHINRA-S1-P0",
        "train_sha256": "abc",
        "total_records": 2,
        "shards": [{
            "canonical_index": 0,
            "path": "canonical/v1/train/train-00000.jsonl",
            "sha256": digest,
            "bytes": len(blob),
            "records": 2,
            "target_tokens": 40,
        }],
    }
    plan = {
        "train_sha256": "abc",
        "updates": [{
            "update_index": 0,
            "record_start": 0,
            "record_end": 2,
            "record_count": 2,
            "target_tokens": 40,
        }],
    }
    verify_authority(
        manifest,
        plan,
        manifest_sha256="m",
        plan_sha256="p",
        expected_manifest_sha256="m",
        expected_plan_sha256="p",
        expected_train_sha256="abc",
    )
    assert verify_shard(shard, manifest["shards"][0]) == 2
    records = [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines()]
    assert take_update(records, plan["updates"][0])[0]["id"] == "a"
    broken = dict(plan["updates"][0])
    broken["target_tokens"] = 1
    try:
        take_update(records, broken)
    except DatasetPinError:
        return
    raise AssertionError("record boundary mismatch accepted")


def test_manifest_order_and_pin_rejected():
    try:
        require_pins(
            "MagistrTheOne/NULLXES-SHINRA-S1-P0",
            "0" * 40,
            "canonical/v1/canonical_manifest.json",
            "canonical/v1/update_plan.json",
        )
    except DatasetPinError:
        pass
    else:
        raise AssertionError("wrong dataset commit accepted")
    manifest = {
        "dataset_repo": "MagistrTheOne/NULLXES-SHINRA-S1-P0",
        "train_sha256": "abc",
        "total_records": 1,
        "shards": [
            {"canonical_index": 1, "path": "b", "records": 1},
            {"canonical_index": 0, "path": "a", "records": 1},
        ],
    }
    plan = {"train_sha256": "abc", "updates": []}
    try:
        verify_authority(
            manifest,
            plan,
            manifest_sha256="m",
            plan_sha256="p",
            expected_manifest_sha256="m",
            expected_plan_sha256="p",
            expected_train_sha256="abc",
        )
    except DatasetPinError:
        return
    raise AssertionError("shuffled manifest order accepted")


def test_inf_gradient_rejected():
    parameter = torch.nn.Parameter(torch.ones(1))
    parameter.grad = torch.tensor([float("inf")])
    try:
        scale_gradients([parameter], 1)
    except ObjectiveError:
        return
    raise AssertionError("inf gradient accepted")


def test_hash_mismatch_rejected(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    path.write_text('{"target_tokens": 1}\n', encoding="utf-8")
    try:
        verify_shard(path, {"sha256": "0" * 64, "bytes": path.stat().st_size, "records": 1, "target_tokens": 1})
    except DatasetPinError:
        return
    raise AssertionError("bad shard hash accepted")


def test_checkpoint_complete_and_latest(tmp_path: Path):
    identity = {
        "dataset_commit": "abc",
        "train_sha256": "t",
        "manifest_sha256": "m",
        "update_plan_sha256": "p",
        "contract_commit": "c",
        "next_update": 1,
    }
    files = {name: b"x" if not name.endswith(".json") else "{}" for name in (
        "model.pt", "optimizer.pt", "rng_python.json", "rng_torch_cpu.pt", "rng_torch_cuda.json", "cursor.json"
    )}
    write_generation(tmp_path, 1, files, identity)
    publish_latest(tmp_path, 1, identity)
    assert load_resume(tmp_path, identity)["generation"] == "gen-000001"
    try:
        publish_latest(tmp_path, 2, identity)
    except CheckpointError:
        return
    raise AssertionError("missing generation was published")


def test_resume_pin_mismatch(tmp_path: Path):
    identity = {
        "dataset_commit": "abc",
        "train_sha256": "t",
        "manifest_sha256": "m",
        "update_plan_sha256": "p",
        "contract_commit": "c",
        "next_update": 1,
    }
    files = {name: b"x" if not name.endswith(".json") else "{}" for name in (
        "model.pt", "optimizer.pt", "rng_python.json", "rng_torch_cpu.pt", "rng_torch_cuda.json", "cursor.json"
    )}
    write_generation(tmp_path, 1, files, identity)
    publish_latest(tmp_path, 1, identity)
    other = dict(identity)
    other["train_sha256"] = "nope"
    try:
        load_resume(tmp_path, other)
    except CheckpointError:
        return
    raise AssertionError("resume pin mismatch accepted")


def _main() -> None:
    import tempfile

    test_prompt_mask_and_eot_supervision()
    test_shift_count_matches_target_tokens()
    test_denominator_scales_once()
    test_overlength_rejected()
    test_double_bos_rejected()
    test_nan_loss_rejected()
    test_manifest_order_and_pin_rejected()
    test_inf_gradient_rejected()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        test_update_plan_replay_and_order(root)
        test_hash_mismatch_rejected(root)
        test_checkpoint_complete_and_latest(root / "ckpt")
        test_resume_pin_mismatch(root / "resume")
    print("STATIC 12 PASS")


if __name__ == "__main__":
    _main()
