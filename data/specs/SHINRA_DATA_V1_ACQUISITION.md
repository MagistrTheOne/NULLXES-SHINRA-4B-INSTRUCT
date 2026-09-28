# SHINRA DATA V1 acquisition

**Status:** contract only. No engine. No network. No adapter. No canary.

```text
governance        != acquisition
acquisition       != materialization
materialization   != canary
canary            != training
```

This file is the git-tracked **boundary** `upstream → raw local artifact → adapter`.  
It does not acquire bytes. It does not convert parquet. It does not stamp a receipt.

Machine schema: [`acquisition.manifest.schema.json`](acquisition.manifest.schema.json)  
Module: `data.data_v1.acquisition`

---

## Layers

```text
GIT SOURCE GOVERNANCE
        |
        v
ACQUISITION CONTRACT          ← this commit
        |
        v
FUTURE ACQUISITION IMPLEMENTATION
        |
        v
LOCAL RAW ACQUISITION ARTIFACT
        |
        v
FUTURE FORMAT ADAPTER
        |
        v
LOCAL JSONL
        |
        v
MATERIALIZATION ABI           ← already GO (c2a8e26)
        |
        v
MATERIALIZED JSONL + RECEIPT
        |
        v
FUTURE CANARY HOOK
```

---

## Namespaces

| root | role |
|---|---|
| `/content/shinra_scratch/data_v1_acquisition` | future **raw** artifacts + sibling manifests |
| `/content/shinra_scratch/data_v1` | materialized canonical JSONL + receipts |
| `/content/shinra_scratch/s0` | S0; forbidden |

This commit does **not** create the production acquisition root.

Validators inject `acquisition_root` (tests: `tmp_path`). The materialization root is not a valid acquisition root.

---

## Governance is the envelope

Identity is taken from [`sources.allowlist.json`](sources.allowlist.json). Unknown `source_id` is rejected. Acquisition must not broaden repository, subset, language, domains, license, or byte caps. Validation does not mutate the allowlist.

Git materialization fields stay:

```text
status = not_materialized
slice_id = null
content_sha256 = null
```

Those fields are **not** acquisition state.

---

## Revision

A revision is either:

- `null` — not frozen; **not** moving HEAD
- a 40-character lowercase git SHA — immutable, only if independently verified

Forbidden labels: `HEAD`, `main`, `master`, `latest`, `current`, `PENDING`, `unknown`.

This commit does not query Hugging Face, GitHub, or HTTP to resolve revisions. It does not replace `null`.

| | governance valid | production-acquisition-ready |
|---|---|---|
| `fineweb-edu-en` revision `null` | yes | **false** |
| `fineweb2-ru` revision `null` | yes | **false** |

Null in governance is valid metadata. It is **not** authorization to acquire. DATA ACQUISITION stays **CLOSED** until a later reviewed source plan freezes an immutable revision.

---

## Raw acquisition artifact

The **raw** artifact is the completed local file produced by a future acquisition layer **before** any adapter and **before** the materialization ABI.

Future formats (metadata enum only; **not parsed here**): `parquet` | `arrow` | `jsonl`.

Identity:

- `source_id`
- upstream `{repository, subset, revision}`
- `artifact.format`
- `artifact.bytes`
- `artifact.raw_content_sha256` = SHA-256 of **exact raw file bytes**
- `artifact.filename` = basename only

Not in the manifest: credentials, URLs, signed URLs, headers, cookies, home paths, absolute source URLs.

```text
raw_content_sha256   SHA of acquired raw bytes
content_sha256       SHA of materialized canonical JSONL (materialization ABI)
```

Different identities. Do not conflate. Do not put `content_sha256` on an acquisition manifest.

---

## Manifest ABI

`schema_version`: `shinra-data-v1-acquisition-manifest`

Sibling file, not embedded in the raw bytes. `status` for handoff is **`complete`**.

`selection.strategy` may be:

- `bounded_bytes` — generic byte cap only
- `single_frozen_file` — one relative posix path + `expected_raw_sha256` matching the raw file

FineWeb-Edu EN uses `single_frozen_file` for `sample/10BT/013_00000.parquet` at revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.

`selection.bound_bytes` is mandatory and in **bytes**. No remote byte-range cuts: the frozen file is acquired whole.

---

## Bounds (bytes)

| gate | bytes |
|---|---:|
| min free disk | `21474836480` (20 GiB) |
| per source | `4294967296` (4 GiB) |
| global completed raw | `8589934592` (8 GiB) |

```text
selection.bound_bytes <= source max_materialized_bytes
artifact.bytes        <= selection.bound_bytes
selection.bound_bytes <= 4294967296
```

No “small slice”, token count, or document count as a storage bound.

Temp files do not count as completed raw.

---

## Complete vs temp

Not acquired:

- `*.part` / `*.tmp` / `*.partial`
- file without sibling complete manifest
- manifest without raw file
- `status != complete`
- SHA or size mismatch

File existence is never status. Manifest existence is never status.

---

## Interrupt / resume (semantics only)

No resume/download code in this commit.

A future implementation may resume a `.part` only if it can prove the same `source_id`, repository, subset, revision, and deterministic selection identity. The partial file stays **incomplete**. No complete manifest, no adapter, no materializer, no canary.

Finalize raw bytes → verify size/SHA → atomically name the raw file → write manifest tmp → fsync → atomically rename manifest. Crash before final manifest: **NOT ACQUIRED**.

---

## Handoff

`validate_acquired_artifact(source_id, manifest_path, acquisition_root, allowlist)` returns metadata only if every contract check passes, including production-ready revision.

It must not: call `materialize()`, convert parquet/arrow, write receipts, edit the allowlist, run Phase B/canary, tokenize, or train.

---

## Adapter boundary

`data/data_v1/materialize.py` stays JSONL-in. Parquet/arrow conversion is a **future adapter**, not this module.
