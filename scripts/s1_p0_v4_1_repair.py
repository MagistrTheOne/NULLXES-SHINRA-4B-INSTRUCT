"""V4.1 projection: drop second SNLI copies, then refill only the deficit."""

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

from data.s1.foundation import sha256_text
from data.s1.p0_amendment import P0_BUDGET, effective_family_targets
from runtime.workspace import resolve_workspace
import s1_build_p0_v3 as builder

REPO = "MagistrTheOne/NULLXES-SHINRA-S1-P0"
V3_REVISION = "3222f76f7d21edfb4ade7ee32511cdd2b65ed122"
PATHS = [f"staging/v2/train/train-{i:05d}.jsonl" for i in range(15)] + [
    f"staging/v4/train/v4-train-{i:05d}.jsonl" for i in range(11)
]
SHARD_ROWS = 60_000


def _open_remote(path: str):
    revision = V3_REVISION if path.startswith("staging/v2/") else None
    local = hf_hub_download(REPO, path, repo_type="dataset", revision=revision)
    return local


def _user_answer(row: dict) -> tuple[str, str]:
    user = next(item["content"] for item in row["messages"] if item["role"] == "user")
    answer = next(item["content"] for item in row["messages"] if item["role"] == "assistant")
    return user, answer


def identify() -> tuple[list[dict], Counter, Counter, Counter, Counter, set[str], set[str]]:
    seen_prompt: dict[str, tuple[str, int]] = {}
    seen_pair: set[str] = set()
    excluded: list[dict] = []
    family, language, source_class, difficulty = Counter(), Counter(), Counter(), Counter()
    kept_prompts: set[str] = set()
    for path in PATHS:
        local = _open_remote(path)
        with open(local, encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                user, answer = _user_answer(row)
                prompt_hash = sha256_text(user)
                pair_hash = sha256_text(user + "\n" + answer)
                if prompt_hash in seen_prompt or pair_hash in seen_pair:
                    kept_path, kept_index = seen_prompt[prompt_hash]
                    if row.get("source") != "snli" or path.startswith("staging/v2/") or not kept_path.startswith("staging/v2/"):
                        raise SystemExit(f"unexpected duplicate {path} {index} source={row.get('source')}")
                    excluded.append({
                        "prompt_hash": prompt_hash,
                        "kept_shard": kept_path,
                        "kept_record_index": kept_index,
                        "excluded_shard": path,
                        "excluded_record_index": index,
                        "source": row.get("source"),
                        "family": row.get("family"),
                        "language": row.get("language"),
                        "difficulty": row.get("difficulty"),
                        "target_tokens": int(row["target_tokens"]),
                    })
                    continue
                seen_prompt[prompt_hash] = (path, index)
                seen_pair.add(pair_hash)
                kept_prompts.add(prompt_hash)
                target = int(row["target_tokens"])
                family[row["family"]] += target
                language[row["language"]] += target
                source_class[row["source_class"]] += target
                difficulty[row["difficulty"]] += target
        Path(local).unlink(missing_ok=True)
        print("identified", path, len(excluded), flush=True)
    return excluded, family, language, source_class, difficulty, kept_prompts, seen_pair


def main() -> None:
    workspace, mode = resolve_workspace()
    if os.environ.get("SHINRA_WORKSPACE_PERSISTENT") != "1" or mode != "BUCKET":
        raise SystemExit("bucket mount required")
    excluded, family, language, source_class, difficulty, kept_prompts, seen_pair = identify()
    if len(excluded) != 4872:
        raise SystemExit(f"excluded {len(excluded)} != 4872")
    if any(item["source"] != "snli" or item["family"] != "S1-06" or item["language"] != "en" for item in excluded):
        raise SystemExit("excluded rows are not EN SNLI S1-06")
    removed = sum(item["target_tokens"] for item in excluded)
    ledger = {"excluded": excluded, "removed_target_tokens": removed}
    ledger_text = json.dumps(ledger, ensure_ascii=False, sort_keys=True)
    ledger_sha = hashlib.sha256(ledger_text.encode("utf-8")).hexdigest()
    out = workspace / "s1-p0" / "v4_1"
    out.mkdir(parents=True, exist_ok=True)
    ledger_path = out / "repair_ledger.json"
    ledger_path.write_text(ledger_text, encoding="utf-8")
    print("POST-EXCLUSION", sum(family.values()), dict(family), dict(language), dict(source_class), "removed", removed, flush=True)
    os.environ["S1_P0_HUB"] = "1"
    job = builder.Build()
    job.seen_prompt = set(kept_prompts)
    job.seen_pair = set(seen_pair)
    job.family_tokens = family
    job.lang_tokens = language
    job.class_tokens = source_class
    job.diff_tokens = difficulty
    job.targets = effective_family_targets(12816)
    before = dict(family)
    job.generate(
        lambda index, lang, level: builder.extract_record(2_000_000 + index, "en", level, "S1-06"),
        "SYNTHETIC",
        "synthetic_S1-06_v41",
        "S1-06",
        ("en",),
    )
    print("REPLACEMENT", len(job.rows), job.family_tokens["S1-06"] - before["S1-06"], flush=True)
    excluded_keys = {(item["excluded_shard"], item["excluded_record_index"]) for item in excluded}
    canon = out / "canonical"
    canon.mkdir(parents=True, exist_ok=True)
    shard_index = 0
    handle = None
    rows_in_shard = 0
    shard_paths: list[Path] = []

    def rotate() -> None:
        nonlocal handle, shard_index, rows_in_shard
        if handle is not None:
            handle.close()
        path = canon / f"train-{shard_index:05d}.jsonl"
        handle = path.open("w", encoding="utf-8", newline="\n")
        shard_paths.append(path)
        shard_index += 1
        rows_in_shard = 0

    rotate()
    for path in PATHS:
        local = _open_remote(path)
        with open(local, encoding="utf-8") as source:
            for index, line in enumerate(source):
                if not line.strip() or (path, index) in excluded_keys:
                    continue
                if rows_in_shard >= SHARD_ROWS:
                    rotate()
                handle.write(line if line.endswith("\n") else line + "\n")
                rows_in_shard += 1
        Path(local).unlink(missing_ok=True)
    for row in job.rows:
        if rows_in_shard >= SHARD_ROWS:
            rotate()
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        rows_in_shard += 1
    if handle is not None:
        handle.close()
    train_hash = hashlib.sha256()
    shard_meta = []
    records = 0
    target_tokens = 0
    for index, path in enumerate(shard_paths):
        blob = path.read_bytes()
        train_hash.update(blob)
        digest = hashlib.sha256(blob).hexdigest()
        shard_records = blob.count(b"\n")
        shard_target = 0
        for line in blob.decode("utf-8").splitlines():
            if line.strip():
                shard_target += int(json.loads(line)["target_tokens"])
        records += shard_records
        target_tokens += shard_target
        shard_meta.append({
            "canonical_index": index,
            "path": f"canonical/v1/train/{path.name}",
            "sha256": digest,
            "bytes": path.stat().st_size,
            "records": shard_records,
            "target_tokens": shard_target,
        })
        print("projected", path.name, shard_records, shard_target, flush=True)
    manifest = {
        "dataset_repo": REPO,
        "contract_commit": "000813d2294b6ec8f2287d4eb46f64c105d0617c",
        "builder_snapshot": "897e7bc1816d70a5fa38b593a50b5a99d50d3497",
        "repair_version": "V4.1",
        "repair_ledger_sha256": ledger_sha,
        "failed_v4_commit": "664cbb5a99b5d111b40af5c204fe8df4080a7024",
        "failed_v4_train_sha256": "bb1d0a7fc947e5eeb7f582ede9cc9ce5a047e2c5624195844304a259d34db07b",
        "seed": 20260924,
        "train_sha256": train_hash.hexdigest(),
        "total_records": records,
        "total_target_tokens": target_tokens,
        "shards": shard_meta,
        "canonicalization": "PROJECTION",
    }
    manifest_path = out / "canonical_manifest.json"
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
    manifest_path.write_text(manifest_text, encoding="utf-8")
    print("REPAIRED", records, target_tokens, train_hash.hexdigest(), ledger_sha, flush=True)
    account = job.account
    prompts: set[str] = set()
    pairs: set[str] = set()
    family_r, language_r, class_r = Counter(), Counter(), Counter()
    identity = 0
    dup = 0
    over = 0
    mismatch = 0
    blacklist = builder.load_blacklist()
    for path in shard_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            user, answer = _user_answer(row)
            sequence, prompt, target, full = account.count(user, answer)
            if (sequence, prompt, target, full) != (row["input_tokens"], row["prompt_tokens"], row["target_tokens"], row["sequence_tokens"]):
                mismatch += 1
            if sequence > 8192:
                over += 1
            prompt_hash = sha256_text(user)
            pair_hash = sha256_text(user + "\n" + answer)
            if prompt_hash in prompts or pair_hash in pairs or prompt_hash in blacklist:
                dup += 1
            prompts.add(prompt_hash)
            pairs.add(pair_hash)
            family_r[row["family"]] += target
            language_r[row["language"]] += target
            class_r[row["source_class"]] += target
            if row["family"] == "S1-10":
                identity += target
    targets = effective_family_targets(identity)
    total = sum(family_r.values()) or 1
    bands = (
        identity == 12816
        and total >= P0_BUDGET
        and dup == 0
        and over == 0
        and mismatch == 0
        and all(family_r[name] >= targets[name] and family_r[name] - targets[name] <= 251 for name in targets)
        and all(0.47 <= language_r[name] / total <= 0.53 for name in ("en", "ru"))
        and 0.50 <= class_r["PUBLIC-DERIVED"] / total <= 0.70
        and 0.20 <= class_r["NATIVE"] / total <= 0.35
        and 0.10 <= class_r["SYNTHETIC"] / total <= 0.25
    )
    updates = []
    cursor = 0
    running = 0
    start = 0
    count = 0
    seen = 0
    if bands:
        for path in shard_paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                target = int(json.loads(line)["target_tokens"])
                running += target
                count += 1
                seen += 1
                if running >= 32768:
                    updates.append({"update_index": len(updates), "record_start": start, "record_end": seen, "record_count": count, "target_tokens": running, "cumulative_target_tokens": sum(item["target_tokens"] for item in updates) + running})
                    start = seen
                    running = 0
                    count = 0
        if count:
            updates.append({"update_index": len(updates), "record_start": start, "record_end": seen, "record_count": count, "target_tokens": running, "cumulative_target_tokens": total})
    plan_path = out / "update_plan.json"
    plan_text = json.dumps({"updates": updates, "train_sha256": train_hash.hexdigest(), "repair_ledger_sha256": ledger_sha}, sort_keys=True)
    plan_path.write_text(plan_text, encoding="utf-8")
    print("GATES", "PASS" if bands else "FAIL", total, dup, mismatch, dict(family_r), dict(language_r), dict(class_r), flush=True)
    if not bands:
        raise SystemExit("repair gates failed")
    api = HfApi()
    operations = [CommitOperationAdd(path_in_repo="staging/v4_1/repair_ledger.json", path_or_fileobj=str(ledger_path))]
    for path in shard_paths:
        operations.append(CommitOperationAdd(path_in_repo=f"canonical/v1/train/{path.name}", path_or_fileobj=str(path)))
    operations.append(CommitOperationAdd(path_in_repo="canonical/v1/canonical_manifest.json", path_or_fileobj=str(manifest_path)))
    operations.append(CommitOperationAdd(path_in_repo="canonical/v1/update_plan.json", path_or_fileobj=str(plan_path)))
    card_path = out / "README.md"
    card_path.write_text(
        "BUILD STATUS: COMPLETE\n\n"
        "V4 finalization detected 4872 duplicated SNLI rows across the V3/V4 resume boundary. "
        "V4.1 removed only the duplicate second occurrences and deterministically restored the resulting target deficit.\n\n"
        f"Contract commit: 000813d2294b6ec8f2287d4eb46f64c105d0617c\n"
        f"Builder snapshot: 897e7bc1816d70a5fa38b593a50b5a99d50d3497\n"
        f"Failed V4 commit: 664cbb5a99b5d111b40af5c204fe8df4080a7024\n"
        f"Repair version: V4.1\n"
        f"Repair ledger SHA256: {ledger_sha}\n"
        f"TRAIN SHA256: {train_hash.hexdigest()}\n"
        f"Manifest SHA256: {hashlib.sha256(manifest_text.encode('utf-8')).hexdigest()}\n"
        f"Update plan SHA256: {hashlib.sha256(plan_text.encode('utf-8')).hexdigest()}\n",
        encoding="utf-8",
    )
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(card_path)))
    commit = api.create_commit(repo_id=REPO, repo_type="dataset", operations=operations, commit_message="v4.1 canonical projection and repair ledger")
    print("REPAIR_COMMIT", getattr(commit, "oid", commit), flush=True)


if __name__ == "__main__":
    main()
