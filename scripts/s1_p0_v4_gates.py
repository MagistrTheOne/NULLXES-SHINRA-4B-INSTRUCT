"""Stream the canonical V3 prefix plus V4 shards and write the final gate report."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_ROOT))
for _name in ("data", "training"):
    if _name not in sys.modules:
        _module = types.ModuleType(_name)
        _module.__path__ = [str(_ROOT / _name)]
        sys.modules[_name] = _module

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from data.s1.blacklist import diagnostic_blacklist
from data.s1.foundation import sha256_text
from data.s1.p0_amendment import P0_BUDGET, effective_family_targets
from runtime.workspace import resolve_workspace
from training.s1_p0_objective import can_attend, gradient_divisor, token_weighted_mean

REPO = "MagistrTheOne/NULLXES-SHINRA-S1-P0"
V3_REVISION = "3222f76f7d21edfb4ade7ee32511cdd2b65ed122"
UPDATE_TARGET = 32_768
TARGETS = effective_family_targets()


def _user_answer(row: dict) -> tuple[str, str]:
    user = next(item["content"] for item in row["messages"] if item["role"] == "user")
    answer = next(item["content"] for item in row["messages"] if item["role"] == "assistant")
    return user, answer


def _iter_remote(api: HfApi, path: str, revision: str):
    local = hf_hub_download(REPO, path, repo_type="dataset", revision=revision)
    with open(local, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield line, json.loads(line)
    Path(local).unlink(missing_ok=True)


def main() -> None:
    workspace, mode = resolve_workspace()
    if os.environ.get("SHINRA_WORKSPACE_PERSISTENT") == "1" and mode != "BUCKET":
        raise SystemExit("bucket mount missing")
    api = HfApi()
    names = api.list_repo_files(REPO, repo_type="dataset")
    v3 = [f"staging/v2/train/train-{index:05d}.jsonl" for index in range(15)]
    v4 = sorted(name for name in names if name.startswith("staging/v4/train/") and name.endswith(".jsonl"))
    order = [(path, V3_REVISION) for path in v3] + [(path, None) for path in v4]
    blacklist = diagnostic_blacklist()
    family = Counter()
    language = Counter()
    source_class = Counter()
    difficulty = Counter()
    drops = Counter()
    seen_prompt: set[str] = set()
    seen_norm: set[str] = set()
    seen_pair: set[str] = set()
    identity_records = 0
    identity_tokens = 0
    records = 0
    input_tokens = 0
    prompt_tokens = 0
    target_tokens = 0
    train_hash = hashlib.sha256()
    shard_meta = []
    updates = []
    update_target = 0
    update_records = 0
    update_index = 0
    first_id = ""
    last_id = ""
    max_sequence = 0
    for path, revision in order:
        digest = hashlib.sha256()
        shard_records = 0
        shard_target = 0
        shard_input = 0
        for raw, row in _iter_remote(api, path, revision or "main"):
            encoded = raw if raw.endswith("\n") else raw + "\n"
            blob = encoded.encode("utf-8")
            digest.update(blob)
            train_hash.update(blob)
            user, answer = _user_answer(row)
            prompt_hash = sha256_text(user)
            pair_hash = sha256_text(user + "\n" + answer)
            if prompt_hash in seen_prompt:
                drops["duplicate-prompt"] += 1
            if prompt_hash in seen_norm:
                drops["normalized-duplicate-prompt"] += 1
            if pair_hash in seen_pair:
                drops["duplicate-prompt-target"] += 1
            if prompt_hash in blacklist:
                drops["frozen-overlap"] += 1
            if row.get("source_split") not in (None, "train"):
                drops["forbidden-split"] += 1
            if int(row["sequence_tokens"]) > 8192:
                drops["overlength"] += 1
            seen_prompt.add(prompt_hash)
            seen_norm.add(prompt_hash)
            seen_pair.add(pair_hash)
            target = int(row["target_tokens"])
            family[row["family"]] += target
            language[row["language"]] += target
            source_class[row["source_class"]] += target
            difficulty[row["difficulty"]] += target
            if row["family"] == "S1-10":
                identity_records += 1
                identity_tokens += target
            records += 1
            input_tokens += int(row["input_tokens"])
            prompt_tokens += int(row["prompt_tokens"])
            target_tokens += target
            shard_records += 1
            shard_target += target
            shard_input += int(row["input_tokens"])
            max_sequence = max(max_sequence, int(row["sequence_tokens"]))
            if not first_id:
                first_id = row["id"]
            last_id = row["id"]
            update_target += target
            update_records += 1
            if update_target >= UPDATE_TARGET:
                updates.append({"index": update_index, "records": update_records, "target_tokens": update_target, "overshoot": update_target - UPDATE_TARGET, "first_record": first_id, "last_record": last_id})
                update_index += 1
                update_target = 0
                update_records = 0
                first_id = ""
        info = api.get_paths_info(REPO, [path], repo_type="dataset", revision=revision)[0]
        shard_meta.append({"path": path, "revision": revision or info.last_commit.oid, "sha256": digest.hexdigest(), "bytes": info.size, "records": shard_records, "input_tokens": shard_input, "target_tokens": shard_target})
        print("scanned", path, shard_records, shard_target, flush=True)
    if update_records:
        updates.append({"index": update_index, "records": update_records, "target_tokens": update_target, "overshoot": max(0, update_target - UPDATE_TARGET), "first_record": first_id, "last_record": last_id})
    total = target_tokens or 1
    family_ok = all(family[name] >= TARGETS[name] for name in TARGETS)
    lang_ok = all(0.47 <= language[name] / total <= 0.53 for name in ("en", "ru"))
    public = source_class["PUBLIC-DERIVED"] / total
    native = source_class["NATIVE"] / total
    synthetic = source_class["SYNTHETIC"] / total
    class_ok = 0.50 <= public <= 0.70 and 0.20 <= native <= 0.35 and 0.10 <= synthetic <= 0.25
    dup_ok = drops["duplicate-prompt"] == 0 and drops["duplicate-prompt-target"] == 0 and drops["frozen-overlap"] == 0 and drops["forbidden-split"] == 0 and drops["overlength"] == 0
    objective = {
        "single": gradient_divisor(1) == 1.0,
        "variable": token_weighted_mean([[1.0], [0.0, 0.0]]) == 1 / 3,
        "isolated": can_attend([0, 0, 1], 2, 1) is False and can_attend([0, 0, 1], 1, 0) is True,
        "denominator": gradient_divisor(updates[0]["target_tokens"]) == float(updates[0]["target_tokens"]),
    }
    passed = target_tokens >= P0_BUDGET and family_ok and lang_ok and class_ok and dup_ok and identity_tokens == 12816 and all(objective.values())
    report = {
        "status": "PASS" if passed else "FAIL",
        "resume_base": V3_REVISION,
        "records": records,
        "v3_shards": 15,
        "v4_shards": len(v4),
        "input_tokens": input_tokens,
        "prompt_tokens": prompt_tokens,
        "target_tokens": target_tokens,
        "budget": P0_BUDGET,
        "overshoot": target_tokens - P0_BUDGET,
        "supervised_fraction": target_tokens / input_tokens if input_tokens else 0,
        "language": dict(language),
        "family": {name: {"target": TARGETS[name], "actual": family[name], "deviation": family[name] - TARGETS[name]} for name in TARGETS},
        "difficulty": dict(difficulty),
        "source_class": dict(source_class),
        "identity": {"records": identity_records, "target_tokens": identity_tokens},
        "drops": dict(drops),
        "max_sequence": max_sequence,
        "train_sha256": train_hash.hexdigest(),
        "updates": {"count": len(updates), "mean_target": sum(item["target_tokens"] for item in updates) / len(updates), "max_overshoot": max(item["overshoot"] for item in updates), "last_target": updates[-1]["target_tokens"]},
        "objective": objective,
        "shards": shard_meta,
    }
    out = workspace / "s1-p0" / "v4" / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "canonical_manifest.json"
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    plan_path = out / "update_plan.json"
    plan_path.write_text(json.dumps(updates, ensure_ascii=False), encoding="utf-8")
    print("GATES", report["status"], report["target_tokens"], report["train_sha256"], flush=True)
    if passed:
        api.create_commit(
            repo_id=REPO,
            repo_type="dataset",
            operations=[
                CommitOperationAdd(path_in_repo="staging/v4/manifests/canonical_manifest.json", path_or_fileobj=str(manifest_path)),
                CommitOperationAdd(path_in_repo="staging/v4/manifests/update_plan.json", path_or_fileobj=str(plan_path)),
            ],
            commit_message="v4 canonical manifest after gates",
        )
        print("manifest-committed", flush=True)


if __name__ == "__main__":
    main()
