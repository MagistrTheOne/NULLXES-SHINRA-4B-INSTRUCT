# SHINRA DATA V1

**Status:** Phase A contract. Not a train recipe. Does not authorize S1.

S0 (`step-00001358`) is mechanics GO, not language GO. `loss≈0.1` on closed synth is a health metric. Language GO is frozen **PROBE V1**.

Four independent contracts:

| path | role |
|---|---|
| [`SHINRA_DATA_V1.md`](SHINRA_DATA_V1.md) | this file: taxonomy, pipeline, phases |
| [`sidecar.schema.json`](sidecar.schema.json) | per-document metadata, never tokenized |
| [`eval/specs/SHINRA_PROBE_V1.md`](../../eval/specs/SHINRA_PROBE_V1.md) | decoding + scorers |
| [`eval/probes/v1/`](../../eval/probes/v1/) | immutable JSONL + SHA manifest |

v1-era `S1_DATA_SPEC.md` / identity replay / `behavior_contract` are not this corpus. `configs/stages/s1_language.yaml` stays the unauthorized synth-v0 spec until phase D.

PT stream:

```text
[BOS] document body [END_OF_TEXT]
```

No chat roles, identity, `<|eot|>` (id 2), ids 4–17, raw URLs in sidecars.

---

## Closed taxonomy

Ingest may not invent domains.

**domain:** `general` | `longform` | `knowledge` | `semantic` | `code` | `math` | `reasoning` | `structured`  
**language:** `en` | `ru`  
**source_type:** `natural` | `synthetic`  
**split:** `train` | `validation`

`reasoning` = pretraining documents with reasoning **structure** (premises, steps, conclusion in prose). Not a `<|reasoning|>` channel and not CoT instruction data.

Probes are **not** a split. They live in `eval/probes/v1/` and never enter the corpus pipeline.

`SHINRA-V2-PRETRAIN-SYNTH-v0` quota: **0%**. New synth (if any) needs a new generator id, a verifier, and must not overlap probe fingerprints.

---

## Sidecar

Unit = **document**, not a packed sequence. Schema: `sidecar.schema.json`.

`document_id` is `sha256:` of canonical NFKC bytes of the body. Exact dedup uses the same hash.

Store `source_id` + `snapshot` + `uri_hash`. **No raw URL.**

Natural: `generator` null. Synthetic: generator required, `*.v0` S0 ids forbidden.

---

## Pipeline (Phase B engine)

Local JSONL/text only. No downloaders. No training.

```text
RAW (local path)
 → normalize (NFKC, strip NUL, reject `<|`)
 → language ID (en|ru script; no fasttext fetch)
 → classify → frozen domain enum
 → sidecar (no URL) + S0 *.v0 ban
 → exact dedup (document_id)
 → near-dedup
 → probe fingerprint firewall
 → body tokens:
      fixture dry-run: whitespace allowed
      real canary: tokenizer required, `add_special_tokens=False`, ids 2/4–17 rejected
 → hard cap ≤ 20_000_000 **tokenizer** tokens (canary)
 → JSON report
```

Pack wrap `[BOS]+body+[END_OF_TEXT=18]` is **not** this phase.

Allowlist: [`SHINRA_DATA_V1_SOURCES.md`](SHINRA_DATA_V1_SOURCES.md) + [`sources.allowlist.json`](sources.allowlist.json) — **empty**, acquisition closed. Disk cap for a future canary: 8 GB raw, 20 GB free. `colab.yaml` 400 GB is not this budget.

```bash
python -m data.data_v1.phase_b --mode fixture --input tests/fixtures/data_v1/phase_b/ok
# real canary (later, after allowlist):
# python -m data.data_v1.phase_b --mode canary --tokenizer tokenizer/artifacts/tokenizer.json --input <local>
```

`scripts/validate_data_v1_phase_a.py` is a **second invocation path** of the Phase A contract, not an independent auditor.

---

## S1 budget (phase D only)

230M new honest tokens. EN/RU 50/50 independently sourced.

| domain | share |
|---|---:|
| general | 38% |
| longform | 15% |
| knowledge | 12% |
| semantic | 10% |
| code | 8% |
| math | 7% |
| reasoning | 5% |
| structured | 3% |
| EN/RU balance buffer | 2% |

`max_steps` must use honest-token pad slack ≥15% (S0: 1221 steps ≈ 18.0M honest vs 20M target).

---

## Phases

```text
A  CONTRACT        0 GPU   taxonomy/schema/probes/SHA/validators
B  DATA CANARY     no train ≤20M ingest + contamination + distribution
C  LEARNING PILOT  20–50M from frozen S0 ckpt, same PROBE V1
D  S1              +230M after explicit authorization
```

Phase C delta: same instrument, two JSON score files. Not impressions.

Colab: `/content` scratch only. Do not Drive-mount the hot path.

---

## Todos

- [x] Sidecar schema (closed enums, no URL, no probe split)
- [x] Probe V1 JSONL + evaluation contract (`use_cache=false`)
- [x] Offline validators + fingerprints
- [x] Partial S0 baseline from the measured session
- [ ] Full S0 probe pass into `eval/baselines/s0-step-00001358/` (GPU, still frozen probes)
- [x] Phase B ingest engine + hard gates + fixture dry-run (no network)
- [x] Source governance schema + empty allowlist; tokenizer required for real canary
- [ ] First authorized source records in `sources.allowlist.json`
- [ ] Real ≤20M **tokenizer-token** canary ingest after allowlist
- [ ] Phase C/D authorization
