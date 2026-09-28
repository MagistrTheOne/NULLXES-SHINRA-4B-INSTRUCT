# SHINRA DATA V1 sources

**Status:** governance frozen. **Allowlist empty.** Acquisition closed.

This file is source law, not a downloader. Phase B engine remains local-path only.

```text
allowed_sources: []
downloaders: none
network: forbidden
acquisition: closed
```

Machine copy: [`sources.allowlist.json`](sources.allowlist.json)  
Schema: [`sources.schema.json`](sources.schema.json)

`configs/stages/s1_language.yaml` is not a source. It stays unauthorized synth-v0 until Phase D.

---

## Token accounting

| run | counter | rule |
|---|---|---|
| fixture dry-run | whitespace **allowed** | engine tests only |
| real corpus canary | **tokenizer required** | `encode(text, add_special_tokens=False)` |
| canary cap | 20_000_000 | tokenizer tokens of the **body**, not packed `[BOS]+body+[18]` |

Whitespace 58 on fixtures is not a corpus budget. A real B that uses whitespace is a failed canary.

Forbidden body ids after tokenize: `2` and `4..17`. Surface `<|` still banned before tokenize.

---

## Record required for every future source

No source enters the allowlist without all of:

| field | rule |
|---|---|
| `source_id` | `^[a-z0-9][a-z0-9._-]{0,127}$` |
| `snapshot` | version / dump id, not a URL |
| `license_id` | SPDX or named license string, not a URL |
| `redistribution` | boolean; `false` cannot ship |
| `source_type` | `natural` \| `synthetic` |
| `languages` | subset of `en`, `ru` |
| `allowed_domains` | subset of the frozen DATA V1 domain enum |
| `acquisition_method` | **only** `local_materialized` |
| `local_materialization_format` | `jsonl` \| `txt` \| `json` |
| `expected_token_range` | `{min, max}` with `max <= 20_000_000` |
| `provenance_hash_strategy` | `content_sha256` \| `source_id_snapshot_path` |

No raw URL in the allowlist. No `hf://`, no Hub id as acquisition method. Materialize elsewhere, then point Phase B at a **local** path.

Synthetic: generator id required at document sidecar; `*.v0` still forbidden.

---

## Disk (Colab G4, this cycle)

`configs/colab.yaml` `disk_ceiling_gb: 400` is **not** Phase B budget. Observed G4 disk in the live session was ~236 GB total, with S0 scratch already occupying space.

Phase B canary disk law:

```text
scratch:     /content/shinra_scratch/data_v1
max raw:     8 GB materialized input
min free:    20 GB after that input (S0 ckpt stays)
Drive:       forbidden on the hot path
```

20M tokenizer tokens of JSONL is hundreds of MB, not tens of GB. If a dump is larger, slice it **before** ingest. Do not fill the disk “to be safe”.

---

## Opening a real canary (later commit)

1. Add one or two source records to `sources.allowlist.json` (`acquisition: local_materialized`).
2. Materialize those dumps locally (outside this engine).
3. `python -m data.data_v1.phase_b --mode canary --tokenizer tokenizer/artifacts/tokenizer.json --input <local>`
4. No training. No S1.

Until `sources` is non-empty, `--mode canary` must fail closed.
