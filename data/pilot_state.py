"""Durable pilot bookkeeping. Parquet is authoritative; checkpoints are caches.

Publication order: complete Parquet -> corpus append journal -> SQLite -> JSON.
Any interrupted publication makes the checkpoint stale and triggers reconstruction.
Never use pickle for checkpoint data. The output and corpus locks require POSIX
(the target environments are Colab/Linux).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasketch import MinHash

from tokenizer.special_tokens import ALL_SPECIAL_TOKENS

from .dedup import DedupIndex, document_minhash, exact_hash

STATE_VERSION = 1


def lang_bucket(code: str) -> str:
    if code in {"en", "eng"}:
        return "en"
    if code == "ru":
        return "ru"
    if code in {"de", "fr", "es", "it", "pt", "nl", "pl", "sv", "cs", "ro", "fi", "hu", "da", "nb", "no"}:
        return "eu"
    return "drop"


def estimate_tokens(row: dict) -> int:
    return max(int(row["n_chars"]) // 4, 1)


def sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, payload: dict) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        sync_dir(path.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextmanager
def build_lock(path: Path):
    import fcntl

    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another pilot builder holds {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def shard_files(output: Path) -> list[Path]:
    files = list(output.glob("clean-*.parquet"))
    for path in files:
        if not re.fullmatch(r"clean-\d+\.parquet", path.name):
            raise ValueError(f"Unrecognized shard name: {path}")
    return sorted(files, key=lambda p: int(p.stem.split("-")[-1]))


def file_stamp(path: Path) -> dict | None:
    if not path.exists():
        return None
    stat = path.stat()
    return {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def manifest(output: Path) -> list[dict]:
    return [file_stamp(path) for path in shard_files(output)]


def publish_shard(output: Path, index: int, rows: list[dict]) -> Path:
    """Hard-link a fully synced temp file: atomic publication WITHOUT clobber."""
    target = output / f"clean-{index:05d}.parquet"
    fd, tmp = tempfile.mkstemp(prefix=".pilot-shard-", dir=output)
    os.close(fd)
    try:
        pq.write_table(pa.Table.from_pylist(rows), tmp, compression="snappy")
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.link(tmp, target)  # EEXIST preserves the old shard even in a race.
        sync_dir(output)
    finally:
        os.unlink(tmp)
    return target


def empty_state() -> dict:
    return {
        "version": STATE_VERSION,
        "datasketch_version": version("datasketch"),
        "bucket_counts": {}, "lang_counts": {}, "source_counts": {},
        "script_counts": {}, "leak_counts": {}, "source_docs": {},
        "total_tokens": 0, "written_docs": 0, "shard_idx": 0,
        "stats": {"seen": 0, "kept": 0, "dropped": 0},
        "source_scan": {}, "scan_counts_complete": True,
        "corpus_chars": 0, "source_stops": {},
    }


def increment(counts: dict, key: str, amount: int = 1) -> None:
    counts[key] = counts.get(key, 0) + amount


def account_document(state: dict, row: dict) -> None:
    tokens = estimate_tokens(row)
    for field, key in (
        ("bucket_counts", row["bucket"]),
        ("lang_counts", lang_bucket(str(row.get("language") or "und"))),
        ("source_counts", row["source"]),
        ("script_counts", str(row.get("script") or "und")),
    ):
        increment(state[field], key, tokens)
    increment(state["source_docs"], row["source"])
    state["total_tokens"] += tokens
    state["written_docs"] += 1
    increment(state["stats"], "kept")
    if row.get("domain") == "code":
        increment(state["stats"], "code_docs")
    if row["bucket"] == "math_stem":
        increment(state["stats"], "math_docs")
    for token in ALL_SPECIAL_TOKENS:
        if token in row["text"]:
            increment(state["leak_counts"], token)
            increment(state["stats"], "special_leaks")


class PilotStore:
    def __init__(self, output: Path, corpus: Path, corpus_limit: int):
        self.output, self.corpus, self.corpus_limit = output, corpus, corpus_limit
        self.state_path = output / "pilot_state.json"
        self.journal = output / "pilot_corpus_journal.json"
        self.db = sqlite3.connect(output / "pilot_index.sqlite3")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY, digest TEXT NOT NULL, signature BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS document_digest ON documents(digest);
            CREATE TABLE IF NOT EXISTS corpus_lines (digest TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT);
        """)
        self.state = empty_state()

    def close(self) -> None:
        self.db.close()

    def recover_corpus_append(self) -> None:
        if not self.journal.exists():
            return
        journal = json.loads(self.journal.read_text(encoding="utf-8"))
        if journal["path"] != str(self.corpus.resolve()):
            raise ValueError("Resume with the original --tokenizer-corpus-dir to recover its append")
        payload = base64.b64decode(journal["payload"], validate=True)
        offset = journal["offset"]
        self.corpus.touch(exist_ok=True)
        with self.corpus.open("r+b") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if not offset <= size <= offset + len(payload):
                raise ValueError("Corpus changed outside the builder during a pending append")
            handle.seek(offset)
            tail = handle.read()
            if not payload.startswith(tail):
                raise ValueError("Corpus tail does not match the pending append; preserving all files")
            handle.seek(0, os.SEEK_END)
            handle.write(payload[len(tail):])
            handle.flush()
            os.fsync(handle.fileno())
        self.journal.unlink()
        sync_dir(self.output)

    def append_corpus(self, rows: list[dict]) -> None:
        chunks = []
        for row in rows:
            remaining = self.corpus_limit - self.state["corpus_chars"]
            if remaining <= 0:
                break
            line = row["text"].replace("\n", " ")[:min(8000, remaining)]
            digest = hashlib.sha256(line.encode("utf-8")).hexdigest()
            added = self.db.execute("INSERT OR IGNORE INTO corpus_lines VALUES (?)", (digest,)).rowcount
            if added:
                chunks.append((line + "\n").encode("utf-8"))
                self.state["corpus_chars"] += len(line)
        if not chunks:
            return
        offset = self.corpus.stat().st_size
        if offset:
            with self.corpus.open("rb") as handle:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    chunks.insert(0, b"\n")
        atomic_json(self.journal, {
            "path": str(self.corpus.resolve()), "offset": offset,
            "payload": base64.b64encode(b"".join(chunks)).decode("ascii"),
        })
        self.recover_corpus_append()

    def checkpoint(self) -> None:
        self.state.update({
            "manifest": manifest(self.output),
            "corpus_stamp": file_stamp(self.corpus),
            "corpus_path": str(self.corpus.resolve()),
        })
        payload = json.dumps(self.state, sort_keys=True)
        self.db.execute("INSERT OR REPLACE INTO metadata VALUES ('state', ?)", (payload,))
        self.db.commit()
        atomic_json(self.state_path, self.state)

    def restore(self, resume: bool) -> str:
        self.recover_corpus_append()
        if resume and self.state_path.exists():
            try:
                state = json.loads(self.state_path.read_text(encoding="utf-8"))
                saved = self.db.execute("SELECT value FROM metadata WHERE key='state'").fetchone()
                valid = (
                    state["version"] == STATE_VERSION
                    and state["datasketch_version"] == version("datasketch")
                    and state["manifest"] == manifest(self.output)
                    and state["corpus_stamp"] == file_stamp(self.corpus)
                    and state["corpus_path"] == str(self.corpus.resolve())
                    and saved is not None and json.loads(saved[0]) == state
                )
                if valid:
                    self.state = state
                    return "checkpoint"
            except (KeyError, ValueError, TypeError):
                pass
        self.reconstruct()
        return "parquet" if resume else "fresh"

    def reconstruct(self) -> None:
        """One-time legacy recovery. Read batches, never modify existing Parquet."""
        self.db.execute("DELETE FROM documents")
        self.db.execute("DELETE FROM corpus_lines")
        self.state = empty_state()
        self.corpus.touch(exist_ok=True)
        with self.corpus.open("rb") as handle:
            for raw in handle:
                line = raw.rstrip(b"\n")
                self.state["corpus_chars"] += len(line.decode("utf-8", errors="replace"))
                digest = hashlib.sha256(line).hexdigest()
                self.db.execute("INSERT OR IGNORE INTO corpus_lines VALUES (?)", (digest,))
        files = shard_files(self.output)
        for path in files:
            print(f"[pilot resume] reconstructing {path.name}", flush=True)
            for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
                rows = batch.to_pylist()
                for row in rows:
                    account_document(self.state, row)
                    self.remember(row)
                self.append_corpus(rows)
        self.state["shard_idx"] = 1 + max((int(p.stem.split("-")[-1]) for p in files), default=-1)
        # Parquet cannot reveal historical rejected/raw rows. Never invent them.
        self.state["scan_counts_complete"] = not files
        self.state["stats"]["seen"] = self.state["written_docs"]
        self.checkpoint()

    def remember(self, row: dict, signature: bytes | None = None) -> None:
        if signature is None:
            signature = document_minhash(row["text"]).hashvalues.astype("<u8").tobytes()
        # Legacy duplicates, if any, remain in Parquet; the index prevents any
        # further writes of their ID or text. It is not used as a document count.
        self.db.execute("INSERT OR IGNORE INTO documents VALUES (?, ?, ?)", (
            str(row["id"]), exact_hash(row["text"]), signature,
        ))


class PilotDedup(DedupIndex):
    """Production exact/MinHash filter with persistent IDs and signatures."""

    def __init__(self, store: PilotStore):
        super().__init__()
        self.store = store
        self.ids: set[str] = set()
        self.last_signature: bytes | None = None
        for doc_id, digest, signature in store.db.execute("SELECT id, digest, signature FROM documents"):
            self.ids.add(doc_id)
            self.exact.add(digest)
            # The default hash scheme differs between datasketch 1.x and 2.x.
            # Checkpoints are invalidated across versions before reaching here.
            mh = MinHash(num_perm=self.num_perm)
            # LSH band keys depend on the dtype's byte representation as well
            # as values (uint32 in datasketch 2's affine32, uint64 in 1.x).
            mh.hashvalues = np.frombuffer(signature, dtype="<u8").astype(mh.hashvalues.dtype, copy=True)
            self.lsh.insert(doc_id, mh)

    def is_duplicate(self, doc_id: str, text: str) -> tuple[bool, str]:
        digest = exact_hash(text)
        if doc_id in self.ids or digest in self.exact:
            self.dropped_exact += 1
            return True, "exact_or_id"
        mh = document_minhash(text, num_perm=self.num_perm, ngram=self.ngram)
        if self.lsh.query(mh):
            self.dropped_near += 1
            return True, "near"
        self.ids.add(doc_id)
        self.exact.add(digest)
        self.lsh.insert(doc_id, mh)
        self.kept += 1
        self.last_signature = mh.hashvalues.astype("<u8").tobytes()
        return False, "keep"
