# SHINRA DATA V1 materialization

**Status:** ABI only. No acquisition. No real canary. No training.

```text
governance     != acquisition
acquisition    != materialization
materialization != canary
materialization != training
```

Git allowlist is the **passport** of an approved source.  
Scratch receipt is the **stamp** of one local batch. They are not the same file.

Machine receipt schema: [`materialization.receipt.schema.json`](materialization.receipt.schema.json)  
Module: `python -m data.data_v1.materialize`

`data/specs/sources.allowlist.json` stays:

```text
materialization.status        = not_materialized
materialization.slice_id      = null
materialization.content_sha256 = null
```

This commit does **not** download FineWeb-Edu or FineWeb2.

---

## State machine

```text
APPROVED SOURCE
      |
      v
LOCAL INPUT EXISTS
      |
      v
MATERIALIZE TO TMP
      |
      v
HASH FINAL BYTES
      |
      v
ATOMIC FINAL JSONL
      |
      v
RECEIPT
      |
      v
VERIFIED MATERIALIZED ARTIFACT
```

Without receipt: **NOT MATERIALIZED**  
Without SHA match: **NOT MATERIALIZED**  
Without governance source: **NOT MATERIALIZED**  
File existence alone is never status.

---

## Layout

Production scratch: `/content/shinra_scratch/data_v1`

```text
/content/shinra_scratch/data_v1/<source_id>/
├── <slice_id>.jsonl
├── <slice_id>.receipt.json
└── crash leftovers (not materialized):
    <source_id>.inprogress.jsonl.tmp
    <slice_id>.receipt.json.tmp
```

Tests inject `scratch_root` (`tmp_path`). They must not write live `/content`.

Forbidden: `/content/shinra_scratch/s0` (equal or descend).

---

## Input (v1)

Already-local **JSONL** only. UTF-8. One JSON object per line. `text` required (string).

`source_id` is an argument and must exist on the governance allowlist with `acquisition_method = local_materialized`.

Rejected: URL, `hf://`, `s3://`, `gs://`, missing file, empty file, directory, non-`.jsonl`.

No parquet/arrow in this ABI. No Hugging Face. No `datasets`. No downloader.

---

## Bytes, SHA, slice_id

Caps (bytes, not “GB”):

| gate | bytes |
|---|---:|
| min free disk | `21474836480` (20 GiB) |
| per source raw | `4294967296` (4 GiB) |
| global completed raw | `8589934592` (8 GiB) |

Global sum counts **only** pairs that pass receipt+JSONL+SHA validation. Temp files do not count.

`content_sha256` = SHA-256 of the **exact final JSONL file bytes** after rename.

Format: `sha256:` + 64 lowercase hex.

```text
slice_id = "<source_id>-" + first 16 hex chars of that SHA-256
```

Example: `fineweb-edu-en-4f6d0c7c899012ab`

No suffixes, timestamps, or UUIDs.

If that final path already exists:

- same SHA + valid receipt + matching metadata → idempotent success
- otherwise → fail closed (no overwrite)

Orphan final JSONL without a valid receipt is **not** auto-completed on retry. Remove it by hand, then rerun.

---

## Atomic write

1. validate governance source  
2. validate local input  
3. disk preflight  
4. mkdir source dir  
5. write `<source_id>.inprogress.jsonl.tmp`  
6. flush + fsync + close  
7. SHA temp bytes → `slice_id`  
8. refuse conflicting destination  
9. atomic rename → `<slice_id>.jsonl`  
10. re-hash FINAL file; mismatch → fail hard  
11. write `<slice_id>.receipt.json.tmp`  
12. flush + fsync + close  
13. atomic rename → receipt  

Crash before JSONL rename: tmp may remain, no receipt, not materialized.  
Crash after JSONL rename, before receipt: JSONL may remain, no receipt, not materialized.  
Crash during receipt tmp: receipt tmp may remain, not materialized.

---

## JSONL serialization ABI

Each output line is `json.dumps(..., ensure_ascii=False, sort_keys=True, separators=(',', ':'))` plus newline.

That **reorders keys and collapses JSON whitespace**. It does **not** rewrite `text` or other field values, inject chat roles, or tokenize.

Not a semantic corpus transform. Sidecar URL policy stays in Phase B ingest.

---

## Receipt

Copied `upstream` must match the governance row **exactly**. If governance `revision` is null, receipt `revision` is null. Do not invent a revision.

Forbidden in receipt: raw URL, tokens, local input path, username, tmp path, HF credentials.

---

## Resolver

`resolve_materialization(source_id, allowlist, scratch_root)` succeeds only for **exactly one** valid pair for that source, with matching SHA, bytes, records, filenames, and upstream.

This ABI does **not** hook `--mode canary`. Production canary stays fail-closed on git `not_materialized`.

---

## Disk truth

Live Colab (governance session): ~236G / 107G used / 129G free.  
`configs/colab.yaml` `disk_ceiling_gb: 400` is not physical capacity.
