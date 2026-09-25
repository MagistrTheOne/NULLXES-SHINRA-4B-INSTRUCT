"""S1-P0 V4 resume. Bucket holds hot state. Dataset Hub gets batched shards."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_ROOT))
import types

for _name in ("data", "training"):
    if _name not in sys.modules:
        _module = types.ModuleType(_name)
        _module.__path__ = [str(_ROOT / _name)]
        sys.modules[_name] = _module

import s1_build_p0_v3 as v3
from runtime.workspace import WorkspaceError, resolve_workspace

REPO = "MagistrTheOne/NULLXES-SHINRA-S1-P0"
V3_REVISION = "3222f76f7d21edfb4ade7ee32511cdd2b65ed122"
PREFIX_RECORDS = 180_000
PREFIX_TARGET = 535_670
PREFIX_SHARDS = 15
PREFIX_FAMILY = {
    "S1-01": 0,
    "S1-02": 140_000,
    "S1-03": 3_498,
    "S1-04": 98_790,
    "S1-05": 261_825,
    "S1-06": 18_741,
    "S1-07": 0,
    "S1-08": 0,
    "S1-09": 0,
    "S1-10": 12_816,
}
PREFIX_LANG = {"en": 256_646, "ru": 279_024}
PREFIX_CLASS = {"NATIVE": 12_816, "PUBLIC-DERIVED": 522_854}
BATCH_SHARDS = 6
COMMIT_BUDGET_PER_HOUR = 8


def _workspace() -> Path:
    path, mode = resolve_workspace()
    if os.environ.get("SHINRA_WORKSPACE_PERSISTENT") == "1" and mode != "BUCKET":
        raise WorkspaceError("persistent bucket mount is required")
    root = path / "s1-p0" / "v4"
    for name in ("state", "state/dedup", "state/source_cursors", "state/counters", "pending/shards", "logs", "manifests"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_prefix(job: v3.Build) -> None:
    state_path = hf_hub_download(
        REPO,
        "staging/v2/state/build_state.json",
        repo_type="dataset",
        revision=V3_REVISION,
    )
    state = _load_json(Path(state_path))
    family = {key: int(state.get("family_tokens", {}).get(key, 0)) for key in PREFIX_FAMILY}
    if family != PREFIX_FAMILY:
        raise SystemExit(f"V3 PREFIX VERIFIED FAIL family {family}")
    lang = {key: int(state.get("language_tokens", {}).get(key, 0)) for key in PREFIX_LANG}
    if lang != PREFIX_LANG:
        raise SystemExit(f"V3 PREFIX VERIFIED FAIL language {lang}")
    shards = state.get("shards") or []
    if len(shards) != PREFIX_SHARDS:
        raise SystemExit(f"V3 PREFIX VERIFIED FAIL shard count {len(shards)}")
    if sum(int(item["records"]) for item in shards) != PREFIX_RECORDS:
        raise SystemExit("V3 PREFIX VERIFIED FAIL record sum")
    if sum(int(item["target_tokens"]) for item in shards) != PREFIX_TARGET:
        raise SystemExit("V3 PREFIX VERIFIED FAIL target sum")
    for item in shards:
        remote = hf_hub_download(
            REPO,
            f"staging/v2/train/{item['filename']}",
            repo_type="dataset",
            revision=V3_REVISION,
        )
        records = 0
        target = 0
        with open(remote, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                records += 1
                target += int(row["target_tokens"])
        if records != int(item["records"]) or target != int(item["target_tokens"]):
            raise SystemExit(f"V3 PREFIX VERIFIED FAIL {item['filename']} records={records} target={target}")
        Path(remote).unlink(missing_ok=True)
    dedup_path = hf_hub_download(
        REPO,
        "staging/v2/state/dedup.json",
        repo_type="dataset",
        revision=V3_REVISION,
    )
    dedup = _load_json(Path(dedup_path))
    job.seen_prompt.update(dedup.get("prompt", []))
    job.seen_pair.update(dedup.get("pair", []))
    job.family_tokens.update(family)
    job.lang_tokens.update(lang)
    job.class_tokens.update(PREFIX_CLASS)
    job.diff_tokens.update(state.get("difficulty_tokens", {}))
    job.identity_tokens = int(state.get("identity_tokens", 0))
    job.drops.update(state.get("drops", {}))
    for name, counter in (state.get("sources") or {}).items():
        job.sources[name].update(counter)
    Path(dedup_path).unlink(missing_ok=True)
    print("V3 PREFIX VERIFIED PASS", PREFIX_RECORDS, PREFIX_TARGET, flush=True)


class V4Sink:
    def __init__(self, job: v3.Build, root: Path) -> None:
        self.job = job
        self.root = root
        self.pending = root / "pending" / "shards"
        self.ready: list[dict] = []
        self.committed: list[dict] = []
        self.next_index = 0
        self.commit_times: list[float] = []
        self.http_429 = 0
        self.api = HfApi()
        state_path = root / "state" / "build_state.json"
        if state_path.is_file():
            saved = _load_json(state_path)
            self.ready = list(saved.get("ready") or [])
            self.committed = list(saved.get("committed") or [])
            self.next_index = int(saved.get("next_index") or 0)
            self.commit_times = list(saved.get("commit_times") or [])
            self.http_429 = int(saved.get("http_429") or 0)
            self.job.family_tokens.update(saved.get("family_tokens") or {})
            self.job.lang_tokens.update(saved.get("language_tokens") or {})
            self.job.class_tokens.update(saved.get("class_tokens") or {})
            self.job.diff_tokens.update(saved.get("difficulty_tokens") or {})
            self.job.drops.update(saved.get("drops") or {})
            self.job.identity_tokens = int(saved.get("identity_tokens") or self.job.identity_tokens)

    def flush(self, force: bool = False) -> None:
        rows = self.job.rows
        if rows:
            name = f"v4-train-{self.next_index:05d}.jsonl"
            path = self.pending / name
            blob = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
            path.write_text(blob, encoding="utf-8")
            self.ready.append({
                "filename": name,
                "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
                "bytes": path.stat().st_size,
                "records": len(rows),
                "input_tokens": sum(row["input_tokens"] for row in rows),
                "target_tokens": sum(row["target_tokens"] for row in rows),
                "status": "PENDING",
            })
            self.job.rows.clear()
            self.next_index += 1
            self.save()
        if len(self.ready) >= BATCH_SHARDS or (force and self.ready):
            self.commit_ready()

    def save(self) -> None:
        job = self.job
        payload = {
            "builder": "s1-p0-v4",
            "seed": v3.SEED,
            "contract_commit": "000813d2294b6ec8f2287d4eb46f64c105d0617c",
            "v3_revision": V3_REVISION,
            "next_index": self.next_index,
            "ready": self.ready,
            "committed": self.committed,
            "commit_times": self.commit_times,
            "http_429": self.http_429,
            "family_tokens": dict(job.family_tokens),
            "language_tokens": dict(job.lang_tokens),
            "class_tokens": dict(job.class_tokens),
            "difficulty_tokens": dict(job.diff_tokens),
            "identity_tokens": job.identity_tokens,
            "drops": dict(job.drops),
            "sources": {name: dict(counter) for name, counter in job.sources.items()},
            "targets": job.targets,
        }
        path = self.root / "state" / "build_state.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
        dedup = self.root / "state" / "dedup" / "index.json"
        dedup_tmp = dedup.with_suffix(".json.tmp")
        dedup_tmp.write_text(
            json.dumps({"prompt": sorted(job.seen_prompt), "pair": sorted(job.seen_pair)}, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(dedup_tmp, dedup)

    def _wait_budget(self) -> None:
        now = time.time()
        self.commit_times = [stamp for stamp in self.commit_times if now - stamp < 3600]
        if len(self.commit_times) >= COMMIT_BUDGET_PER_HOUR:
            sleep_for = 3600 - (now - self.commit_times[0]) + 1
            print("commit-budget-wait", int(sleep_for), flush=True)
            time.sleep(max(1, sleep_for))

    def commit_ready(self) -> None:
        if not self.ready:
            return
        self._wait_budget()
        operations = []
        for item in self.ready:
            item["status"] = "UPLOADING"
            operations.append(
                CommitOperationAdd(
                    path_in_repo=f"staging/v4/train/{item['filename']}",
                    path_or_fileobj=str(self.pending / item["filename"]),
                )
            )
        self.save()
        delay = 30
        for attempt in range(8):
            try:
                commit = self.api.create_commit(
                    repo_id=REPO,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=f"v4 batch {self.ready[0]['filename']}..{self.ready[-1]['filename']}",
                )
                break
            except Exception as exc:
                response = getattr(exc, "response", None)
                status = getattr(response, "status_code", None)
                if status != 429:
                    raise
                self.http_429 += 1
                retry_after = 0
                if response is not None:
                    retry_after = int(response.headers.get("Retry-After") or 0)
                wait = retry_after or min(1800, delay + random.randint(0, 15))
                print("429-backoff", wait, attempt, flush=True)
                self.save()
                time.sleep(wait)
                delay = min(1800, delay * 2)
        else:
            self.save()
            raise SystemExit("429 remained after bounded retries")
        oid = getattr(commit, "oid", None) or getattr(commit, "commit_url", "")
        for item in self.ready:
            info = self.api.get_paths_info(REPO, [f"staging/v4/train/{item['filename']}"], repo_type="dataset")[0]
            if info.size != item["bytes"]:
                raise SystemExit(f"remote size mismatch {item['filename']}")
            item["status"] = "COMMITTED"
            item["commit"] = oid
        self.commit_times.append(time.time())
        self.committed.extend(self.ready)
        self.ready = []
        self.save()
        print("hub-batch", oid, "commits-hour", len(self.commit_times), "429", self.http_429, flush=True)


def prepare(job: v3.Build) -> None:
    root = _workspace()
    if not (root / "state" / "build_state.json").is_file():
        verify_prefix(job)
    job.sink = V4Sink(job, root)


def main() -> None:
    os.environ["S1_P0_V4"] = "1"
    os.environ["HF_HOME"] = "/tmp/hf"
    os.environ["HF_HUB_CACHE"] = "/tmp/hf/hub"
    os.environ["HF_DATASETS_CACHE"] = "/tmp/hf/datasets"
    v3.Build._v4_prepare = prepare
    job = v3.build()
    job._maybe_flush(force=True)
    print("v4-continue-done", dict(job.family_tokens), dict(job.class_tokens), flush=True)


if __name__ == "__main__":
    main()
