# SHINRA probe V1

**Status:** frozen evaluation contract. Immutable artifact `eval/probes/v1/`.

Do not edit JSONL or `manifest.json` in place. A bad item means **probe-v1.1 or V2**, so the S0 baseline stays reproducible.

Probes live **outside** the corpus pipeline. They are not `split: probe`.

---

## Instrument

```text
S0 checkpoint ──► PROBE V1 ──► baseline.json
pilot checkpoint ──► PROBE V1 ──► pilot.json
                                      │
                                      ▼
                                    DELTA
```

Same files, same tokenizer ids, same decoding, same scorers. Not “the text looks better”.

---

## Frozen decoding

From `eval/probes/v1/manifest.json`:

| field | value |
|---|---|
| `do_sample` | `false` |
| `use_cache` | `false` |
| `max_new_tokens` cap | `128` (record may be lower) |
| wrap | `[BOS] + encode(prompt, add_special_tokens=False)` |
| stop | `document_end_id=18` |
| `eot_id=2` | must not be used as document stop |

`use_cache=false` is part of this contract. The KV-mask `generate()` path is known broken; it must not silently change scores.

Checkpoint named in the manifest: `s0/step-00001358`.

---

## Scorers

Capability GO uses only measurable scorers. Free continuation is logged, not a vibe gate.

| scorer | `expected` | pass |
|---|---|---|
| `numeric_exact` | number | first number in generation equals expected |
| `choice_exact` | `A`/`B`/`C` | first standalone A/B/C |
| `text_exact` | string | stripped equality |
| `text_contains` | string | case-insensitive substring |
| `json_parse` | null | first `{...}` or `[...]` in prompt+generation parses |
| `forbidden_lexicon` | null | generation has no S0 closed-lexicon spans |
| `identity_absent` | null | no identity needles |
| `chat_absent` | null | no chat/instruction needles |
| `generation_log` | null | stored only; not a GO bit |

S0 closed lexicon (collapse fail on OOD/completion): names and stock phrases from `language_core.v0` / `structured.v0` (Mira/Alina/Nadir, “near the door”, `"ok": true`, …). Listed in `data/data_v1/phase_a.py` as `S0_CLOSED_LEXICON`.

---

## Files

`eval/probes/v1/*.jsonl` plus `manifest.json` with:

- per-file SHA256
- `probe_bundle_sha256`
- `manifest_sha256`

After freeze, hash mismatch ⇒ new version, not a silent edit.

---

## Contamination firewall

After freeze, build a fingerprint set from every prompt, `expected`, concatenations, and 40-char spans.

Ingest **must reject** train/validation documents that contain:

- exact probe text
- normalized exact match
- prompt+answer concatenation
- a long matching span (≥40 normalized chars)

Especially math/code: a generator can accidentally reproduce a tiny probe.

---

## Baseline

`eval/baselines/s0-step-00001358/`

| file | role |
|---|---|
| `generations.jsonl` | prompt/generation pairs from the frozen instrument |
| `scores.json` | scorer outputs |
| `run_manifest.json` | checkpoint, decoding, coverage |

Ad-hoc S0 session outputs are stored as coverage=`partial`. A full S0 pass on this probe bundle (still `use_cache=false`) fills the remaining ids **without changing probe files**.

---

## Phases

```text
A  CONTRACT     0 GPU    this bundle + sidecar schema + validators
B  DATA CANARY  no train ≤20M ingest report + contamination
C  LEARNING PILOT 20–50M from frozen S0 ckpt, same PROBE V1
D  S1 +230M     explicit authorization only
```

`configs/stages/s1_language.yaml` stays untouched until D.
