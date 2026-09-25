"""Finalize the already built V4 corpus. Does not ingest or rewrite shards."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
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
import s1_build_p0_v3 as builder

REPO = "MagistrTheOne/NULLXES-SHINRA-S1-P0"
V3_REVISION = "3222f76f7d21edfb4ade7ee32511cdd2b65ed122"
BUILDER_SNAPSHOT = "897e7bc1816d70a5fa38b593a50b5a99d50d3497"
CONTRACT = "000813d2294b6ec8f2287d4eb46f64c105d0617c"
SEED = 20260924
UPDATE_TARGET = 32_768
EXPECTED_IDENTITY = 12816
EXPECTED = {
    "S1-01": 392713,
    "S1-02": 392713,
    "S1-03": 314171,
    "S1-04": 314171,
    "S1-05": 261809,
    "S1-06": 314171,
    "S1-07": 261809,
    "S1-08": 130904,
    "S1-09": 104723,
    "S1-10": 12816,
}
V3_PATHS = [f"staging/v2/train/train-{index:05d}.jsonl" for index in range(15)]
V4_PATHS = [f"staging/v4/train/v4-train-{index:05d}.jsonl" for index in range(11)]


def _quantile(values: list[int], q: float) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def _user_answer(row: dict) -> tuple[str, str]:
    user = next(item["content"] for item in row["messages"] if item["role"] == "user")
    answer = next(item["content"] for item in row["messages"] if item["role"] == "assistant")
    return user, answer


def reconcile(api: HfApi, workspace: Path) -> tuple[dict, list[dict], list[str]]:
    state_path = workspace / "s1-p0" / "v4" / "state" / "build_state.json"
    if not state_path.is_file():
        raise SystemExit("bucket build_state.json missing")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    committed = list(state.get("committed") or [])
    ready = list(state.get("ready") or [])
    blockers = []
    if ready:
        blockers.append(f"bucket has {len(ready)} uncommitted ready shards")
    remote = set(api.list_repo_files(REPO, repo_type="dataset"))
    by_name = {item["filename"]: item for item in committed}
    table = []
    for path in V3_PATHS + V4_PATHS:
        name = path.rsplit("/", 1)[-1]
        exists = path in remote
        info = api.get_paths_info(REPO, [path], repo_type="dataset")[0] if exists else None
        bucket = by_name.get(name)
        row = {
            "path": path,
            "bucket_status": None if path.startswith("staging/v2/") else (bucket or {}).get("status"),
            "remote": exists,
            "bytes": None if info is None else info.size,
            "sha256": None if bucket is None else bucket.get("sha256"),
            "records": None if bucket is None else bucket.get("records"),
            "target_tokens": None if bucket is None else bucket.get("target_tokens"),
        }
        if not exists:
            blockers.append(f"missing remote {path}")
        if path.startswith("staging/v4/"):
            if bucket is None:
                blockers.append(f"remote shard without bucket commit mark {name}")
            elif bucket.get("status") != "COMMITTED":
                blockers.append(f"{name} bucket status {bucket.get('status')}")
            elif info is not None and bucket.get("bytes") != info.size:
                blockers.append(f"{name} size bucket {bucket.get('bytes')} remote {info.size}")
        table.append(row)
    extra = sorted(path for path in remote if path.startswith("staging/v4/train/") and path not in V4_PATHS)
    if extra:
        blockers.append(f"unexpected v4 shards {extra}")
    return state, table, blockers


def scan(api: HfApi, account: builder.Account, blacklist: set[str], registry: dict, overlap: dict[str, set[str]]) -> dict:
    approved = {item["source_id"] for item in registry["sources"] if item.get("license_status") == "APPROVED"}
    family = Counter()
    language = Counter()
    source_class = Counter()
    difficulty = Counter()
    sources = Counter()
    prompts: set[str] = set()
    pairs: set[str] = set()
    identity_prompts: set[str] = set()
    drops = Counter()
    violations = Counter()
    records = 0
    input_tokens = 0
    prompt_tokens = 0
    target_tokens = 0
    identity_records = 0
    identity_tokens = 0
    train_hash = hashlib.sha256()
    shard_meta = []
    target_lengths: list[int] = []
    sequence_lengths: list[int] = []
    family_max = Counter()
    cursor = 0
    updates = []
    update_target = 0
    update_records = 0
    update_index = 0
    record_start = 0
    for path in V3_PATHS + V4_PATHS:
        revision = V3_REVISION if path.startswith("staging/v2/") else None
        local = hf_hub_download(REPO, path, repo_type="dataset", revision=revision)
        digest = hashlib.sha256()
        shard_records = 0
        shard_target = 0
        with open(local, "rb") as handle:
            raw_file = handle.read()
        digest.update(raw_file)
        train_hash.update(raw_file)
        for line in raw_file.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            user, answer = _user_answer(row)
            sequence, prompt, target, full = account.count(user, answer)
            if (sequence, prompt, target, full) != (
                int(row["input_tokens"]),
                int(row["prompt_tokens"]),
                int(row["target_tokens"]),
                int(row["sequence_tokens"]),
            ):
                violations["objective-mismatch"] += 1
            if target != full - prompt:
                violations["shift-mismatch"] += 1
            prompt_hash = sha256_text(user)
            pair_hash = sha256_text(user + "\n" + answer)
            if prompt_hash in prompts:
                drops["duplicate-prompt"] += 1
            if pair_hash in pairs:
                drops["duplicate-prompt-target"] += 1
            if prompt_hash in blacklist:
                drops["frozen-overlap"] += 1
            for name, hashes in overlap.items():
                if prompt_hash in hashes:
                    drops[name] += 1
            prompts.add(prompt_hash)
            pairs.add(pair_hash)
            if row.get("source_split") not in (None, "train"):
                violations["forbidden-split"] += 1
            source = str(row.get("source"))
            if source not in approved:
                violations["unapproved-source"] += 1
            if not row.get("metadata", {}).get("license"):
                violations["unknown-license"] += 1
            lowered = source.lower()
            if "qqp" in lowered or "paws-x" in lowered or "xnli" in lowered:
                violations["forbidden-source"] += 1
            if source == "snli" and "vg_" in str(row.get("provenance_id")):
                violations["snli-provenance"] += 1
            if int(row["sequence_tokens"]) > 8192 or sequence > 8192:
                violations["overlength"] += 1
            family[row["family"]] += target
            language[row["language"]] += target
            source_class[row["source_class"]] += target
            difficulty[row["difficulty"]] += target
            sources[source] += target
            family_max[row["family"]] = max(family_max[row["family"]], target)
            if row["family"] == "S1-10":
                identity_records += 1
                identity_tokens += target
                if prompt_hash in identity_prompts:
                    violations["identity-duplicate"] += 1
                identity_prompts.add(prompt_hash)
            records += 1
            input_tokens += int(row["input_tokens"])
            prompt_tokens += int(row["prompt_tokens"])
            target_tokens += target
            shard_records += 1
            shard_target += target
            target_lengths.append(target)
            sequence_lengths.append(sequence)
            update_target += target
            update_records += 1
            cursor += 1
            if update_target >= UPDATE_TARGET:
                updates.append({
                    "update_index": update_index,
                    "record_start": record_start,
                    "record_end": cursor,
                    "record_count": update_records,
                    "target_tokens": update_target,
                    "cumulative_target_tokens": target_tokens,
                })
                update_index += 1
                update_target = 0
                update_records = 0
                record_start = cursor
        info = api.get_paths_info(REPO, [path], repo_type="dataset", revision=revision)[0]
        shard_meta.append({
            "canonical_index": len(shard_meta),
            "path": path,
            "sha256": digest.hexdigest(),
            "bytes": info.size,
            "records": shard_records,
            "target_tokens": shard_target,
            "revision": revision or (info.last_commit.oid if info.last_commit is not None else api.repo_info(REPO, repo_type="dataset").sha),
        })
        Path(local).unlink(missing_ok=True)
        print("scanned", path, shard_records, shard_target, flush=True)
    if update_records:
        updates.append({
            "update_index": update_index,
            "record_start": record_start,
            "record_end": cursor,
            "record_count": update_records,
            "target_tokens": update_target,
            "cumulative_target_tokens": target_tokens,
        })
    return {
        "records": records,
        "input_tokens": input_tokens,
        "prompt_tokens": prompt_tokens,
        "target_tokens": target_tokens,
        "family": family,
        "language": language,
        "source_class": source_class,
        "difficulty": difficulty,
        "sources": sources,
        "drops": drops,
        "violations": violations,
        "identity_records": identity_records,
        "identity_tokens": identity_tokens,
        "train_sha256": train_hash.hexdigest(),
        "shards": shard_meta,
        "target_lengths": target_lengths,
        "sequence_lengths": sequence_lengths,
        "family_max": family_max,
        "updates": updates,
        "duplicate_prompts": drops["duplicate-prompt"],
    }


def main() -> None:
    workspace, mode = resolve_workspace()
    if os.environ.get("SHINRA_WORKSPACE_PERSISTENT") != "1" or mode != "BUCKET":
        raise SystemExit("bucket mount required")
    api = HfApi()
    state, table, blockers = reconcile(api, workspace)
    print("RECONCILE", "PASS" if not blockers else "FAIL", blockers, flush=True)
    account = builder.Account()
    bos = account.tokenizer.token_to_id("<|bos|>")
    eot = account.tokenizer.token_to_id("<|eot|>")
    pad = account.tokenizer.token_to_id("<|pad|>")
    vocab = account.tokenizer.get_vocab_size()
    tokenizer_ok = (bos, eot, pad, vocab) == (1, 2, 3, 131072)
    registry = json.loads((_ROOT / "data" / "s1" / "source_registry.json").read_text(encoding="utf-8"))
    blacklist = builder.load_blacklist()
    frozen_payload = json.loads((_ROOT / "data" / "s1" / "frozen_diagnostic_hashes.json").read_text(encoding="utf-8"))
    dev_payload = json.loads((_ROOT / "data" / "s1" / "p0_dev" / "dev_prompt_hashes.json").read_text(encoding="utf-8"))
    from data.s1.blacklist import QA_PROMPTS
    overlap = {
        "s05": set(frozen_payload["sets"]["s05"]["prompt_sha256"]),
        "s07": set(frozen_payload["sets"]["s07"]["prompt_sha256"]),
        "qa12": {sha256_text(prompt) for prompt in QA_PROMPTS},
        "s1dev": set(dev_payload["prompt_sha256"]),
    }
    first = scan(api, account, blacklist, registry, overlap)
    second = scan(api, account, blacklist, registry, overlap)
    deterministic = first["train_sha256"] == second["train_sha256"] and first["records"] == second["records"] and first["target_tokens"] == second["target_tokens"]
    identity_tokens = first["identity_tokens"]
    targets = effective_family_targets(identity_tokens)
    target_match = targets == EXPECTED and identity_tokens == EXPECTED_IDENTITY
    total = first["target_tokens"] or 1
    family_fail = []
    for name, budget in EXPECTED.items():
        actual = first["family"][name]
        if actual < budget or actual - budget > first["family_max"][name]:
            family_fail.append(name)
    overshoot = total - P0_BUDGET
    max_target = max(first["target_lengths"] or [0])
    budget_ok = total >= P0_BUDGET and 0 <= overshoot <= max_target
    lang_ok = all(0.47 <= first["language"][name] / total <= 0.53 for name in ("en", "ru"))
    public = first["source_class"]["PUBLIC-DERIVED"] / total
    native = first["source_class"]["NATIVE"] / total
    synthetic = first["source_class"]["SYNTHETIC"] / total
    class_ok = 0.50 <= public <= 0.70 and 0.20 <= native <= 0.35 and 0.10 <= synthetic <= 0.25
    dup_ok = first["drops"]["duplicate-prompt"] == 0 and first["drops"]["duplicate-prompt-target"] == 0 and first["drops"]["frozen-overlap"] == 0
    violation_ok = sum(first["violations"].values()) == 0
    sequence_ok = sum(1 for value in first["sequence_lengths"] if value > 8192) == 0
    objective_ok = first["violations"]["objective-mismatch"] == 0 and first["violations"]["shift-mismatch"] == 0
    content_ok = (not blockers) and tokenizer_ok and target_match and not family_fail and budget_ok and lang_ok and class_ok and dup_ok and violation_ok and sequence_ok and objective_ok and deterministic
    manifest = {
        "dataset_repo": REPO,
        "builder_snapshot": BUILDER_SNAPSHOT,
        "contract_commit": CONTRACT,
        "seed": SEED,
        "resume_base": V3_REVISION,
        "hash_method": "sha256 of concatenated canonical JSONL bytes in manifest shard order",
        "total_records": first["records"],
        "total_target_tokens": first["target_tokens"],
        "train_sha256": first["train_sha256"],
        "shards": first["shards"],
        "reconciliation": table,
        "bucket_family_tokens": state.get("family_tokens"),
        "bucket_language_tokens": state.get("language_tokens"),
        "bucket_class_tokens": state.get("class_tokens"),
        "bucket_drops": state.get("drops"),
        "bucket_sources": state.get("sources"),
    }
    plan = {
        "dataset_commit_input": api.repo_info(REPO, repo_type="dataset").sha,
        "train_sha256": first["train_sha256"],
        "contract_commit": CONTRACT,
        "seed": SEED,
        "updates": first["updates"] if content_ok else [],
    }
    out = workspace / "s1-p0" / "v4" / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "canonical_manifest.json"
    plan_path = out / "update_plan.json"
    report_path = out / "final_report.json"
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
    plan_text = json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True)
    manifest_path.write_text(manifest_text, encoding="utf-8")
    plan_path.write_text(plan_text, encoding="utf-8")
    lengths = first["target_lengths"]
    sequences = first["sequence_lengths"]
    report = {
        "status": "PASS" if content_ok else "FAIL",
        "blockers": blockers + family_fail,
        "violations": dict(first["violations"]),
        "drops": dict(first["drops"]),
        "identity_tokens": identity_tokens,
        "targets": targets,
        "family": {name: {"target": EXPECTED[name], "actual": first["family"][name], "deviation": first["family"][name] - EXPECTED[name]} for name in EXPECTED},
        "language": dict(first["language"]),
        "source_class": dict(first["source_class"]),
        "difficulty": dict(first["difficulty"]),
        "sources": dict(first["sources"]),
        "records": first["records"],
        "input_tokens": first["input_tokens"],
        "prompt_tokens": first["prompt_tokens"],
        "target_tokens": first["target_tokens"],
        "overshoot": overshoot,
        "train_sha256": first["train_sha256"],
        "manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
        "update_plan_sha256": hashlib.sha256(plan_text.encode("utf-8")).hexdigest(),
        "deterministic": deterministic,
        "tokenizer_ok": tokenizer_ok,
        "updates": {
            "count": len(first["updates"]) if content_ok else 0,
            "mean": statistics.mean(item["target_tokens"] for item in first["updates"]) if content_ok and first["updates"] else 0,
            "max_overshoot": max(max(0, item["target_tokens"] - UPDATE_TARGET) for item in first["updates"]) if content_ok and first["updates"] else 0,
            "last": first["updates"][-1]["target_tokens"] if content_ok and first["updates"] else 0,
        },
        "target_length": {
            "min": min(lengths) if lengths else 0,
            "p50": _quantile(lengths, 0.50),
            "mean": statistics.mean(lengths) if lengths else 0,
            "p95": _quantile(lengths, 0.95),
            "p99": _quantile(lengths, 0.99),
            "max": max(lengths) if lengths else 0,
        },
        "sequence_length": {
            "max": max(sequences) if sequences else 0,
            "p50": _quantile(sequences, 0.50),
            "p95": _quantile(sequences, 0.95),
            "p99": _quantile(sequences, 0.99),
            "over_8192": sum(1 for value in sequences if value > 8192),
        },
        "identity_records": first["identity_records"],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print("GATES", report["status"], report["target_tokens"], report["train_sha256"], flush=True)
    operations = [
        CommitOperationAdd(path_in_repo="staging/v4/manifests/final_report.json", path_or_fileobj=str(report_path)),
    ]
    if content_ok:
        card = (
            "---\nlicense: other\npretty_name: NULLXES SHINRA S1-P0\n---\n\n"
            "# NULLXES SHINRA S1-P0\n\nBUILD STATUS: COMPLETE\n\n"
            f"Contract commit: {CONTRACT}\nBuilder snapshot: {BUILDER_SNAPSHOT}\n"
            f"TRAIN SHA256: {report['train_sha256']}\n"
            f"Manifest SHA256: {report['manifest_sha256']}\n"
            f"Update plan SHA256: {report['update_plan_sha256']}\n"
        )
        card_path = out / "README.md"
        card_path.write_text(card, encoding="utf-8")
        operations.extend([
            CommitOperationAdd(path_in_repo="staging/v4/manifests/canonical_manifest.json", path_or_fileobj=str(manifest_path)),
            CommitOperationAdd(path_in_repo="staging/v4/manifests/update_plan.json", path_or_fileobj=str(plan_path)),
            CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(card_path)),
        ])
    commit = api.create_commit(repo_id=REPO, repo_type="dataset", operations=operations, commit_message="v4 finalization artifacts")
    print("FINAL_COMMIT", getattr(commit, "oid", commit), flush=True)


if __name__ == "__main__":
    main()
