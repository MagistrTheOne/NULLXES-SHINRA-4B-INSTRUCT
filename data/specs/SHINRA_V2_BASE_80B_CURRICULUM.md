# SHINRA V2 BASE-80B curriculum

Status: CONTRACT
Corpus train through S2 this cycle: **100% synthetic** (`SHINRA-V2-PRETRAIN-SYNTH-v0`)
Natural mix after 1B: **spec only** (not downloaded here)
Hardware: Colab G4 profile [`configs/colab.yaml`](../../configs/colab.yaml) — not a train recipe
Packer: `shinra-v2-pack.v1`
Init: random. No foreign checkpoints / Hub v1 / S0.4.1 weights.
Attention on G4: **SDPA** (SM120). Not FA3. Not Hub overwrite.

Target object (line, not this cycle): **SHINRA V2 BASE-80B**

| field | value |
|---|---|
| params | 3,969,056,256 |
| tokenizer | 131072 |
| context now | 2048 |
| later context | 2048 → 4096 → 8192 → long |
| pretrain scale | ~80B cumulative tokens |
| disk ceiling | 400 GB rolling drum |

Chinchilla-order: `3.969e9 × 20 ≈ 79.4B`. Not sacred. Order is right. Do not materialize 80B on disk at once.

## Stages

Cumulative targets. S0–S2 **this cycle** train 100% synth. Intended future mix is recorded, not executed.

| stage | cumulative | new tokens | this cycle data | intended mix later |
|---|---:|---:|---|---|
| S0 bring-up | 20M | 20M | 100% synth | 100% synth |
| S1 language ignition | 250M | 230M | 100% synth | 90% synth / 10% curated natural |
| S2 semantic core | 1B | 750M | 100% synth | 70% synth / 30% natural |
| S3 composition | 4B | 3B | spec only | 35% synth / 65% natural |
| S4 foundation | 12B | 8B | spec only | 15% synth / 85% natural |
| S5 general base | 30B | 18B | spec only | mixed natural + small synth |
| S6 mature base | 60B | 30B | spec only | mixed natural + small synth |
| S7 final base | 80B | 20B | spec only | quality-heavy anneal |

S5–S7 intended buckets (not frozen sources): 50–60% high-quality text, 10–15% scientific/technical, 10–15% code, 5–10% math, 5–10% RU, small controlled synth. Not FineWeb-mandatory.

## Rolling disk

Packed `uint32` ids: 1 token = 4 bytes. Active window ~20–25B tokens ≈ 80–100 GB, not 80B at once.

```
corpus/s0|s1|s2/shard-XXXXX.bin + .meta.json
corpus/heldout/
run/ledger.jsonl
run/metrics.jsonl
run/status.json
```

Controller: generate → pack → SHA → train → mark consumed → delete scratch → next. Abort if usage exceeds `disk_ceiling_gb`.

## Gates

**S0:** loss moved, finite grads, pack has no EOT=2 / ids 4–17. Not “language”.

**S1:** grammar/morphology probes not dead.

**S2 / 1B GO/NO-GO:** held-out CE; EN/RU; negation; syntax; relation direction; coref; multi-sentence continuation; repetition; entropy; END_OF_TEXT behavior; generate without identity; grad/activation finite.

If 1B clean tokens still yield absolute garbage, do not use “need more data” as cover. Cut architecture / init / optimizer / pipeline. Do not open S3.

## Out of this cycle

FineWeb dump, 80B materialize, Hub INSTRUCT replace, tokenizer retrain, FA3, identity replay, S0.4.1 resume, turning `colab.yaml` into an 80B train recipe.
