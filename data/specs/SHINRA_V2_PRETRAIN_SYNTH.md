# SHINRA v2 pretrain synthetic corpus

Status: SPEC FREEZE (v0)
Corpus id: `SHINRA-V2-PRETRAIN-SYNTH-v0`
License field: `nullxes-synthetic-v0`
Packer: `shinra-v2-pack.v1` (`docs/tokenizer_pretrain_contract.md`)

This is **causal document text**, not SFT. Records are plain documents. The language model never sees `semantics`.

## Purpose

Programmatic, seeded documents for a later 20–50M token **probe** run on random-init SHINRA v2.

This cycle specifies generators and ships CPU fixtures (about 10^3–10^4 documents). It does **not** emit 50M tokens, does not mix FineWeb, and does not use an LLM as the primary generator.

50M tokens is not Chinchilla-scale language. If later probes stay dead, the curriculum failed; that is not a reason to pretend web mix was required in v0.

## Five layers (plain documents)

| layer | content | not allowed |
|---|---|---|
| `language_core` | EN/RU ~50/50 sentences, morphology, punctuation | QA, chat roles |
| `semantic_primitives` | relation, negation, coref, temporal, comparison as prose paraphrases of a hidden graph | instruction templates |
| `compositional` | short paragraphs, controlled state changes | identity / “I am SHINRA” |
| `knowledge_shaped` | invented entities (Lerna-7, Varek, K-12) | Wikipedia dump |
| `structured` | small code, arithmetic, lists, JSON-like | FineWeb |

Language mix inside each layer: English 50%, Russian 50% (record count, not Unigram tokens). Tolerance for fixtures: exact half when `per_layer` is even.

## Record schema

Each JSONL line:

| field | type | notes |
|---|---|---|
| `text` | string | LM document. No special-token strings. No chat. |
| `layer` | string | one of the five layer ids |
| `language` | string | `en` or `ru` |
| `generator` | string | function id, e.g. `language_core.v0` |
| `seed` | int | corpus seed mixed with layer/language/index |
| `semantics` | object | hidden graph / labels. **Never concatenated into `text`.** |

`semantics` is for audits and later probes. Packer reads `text` only.

## Manifest schema

One manifest per `(family, language)` shard. JSON Schema: `data/specs/shinra_v2_pretrain_synth.manifest.schema.json`.

Required fields:

| field | type | notes |
|---|---|---|
| `corpus_id` | string | `SHINRA-V2-PRETRAIN-SYNTH-v0` |
| `family` | string | layer id |
| `language` | string | `en` or `ru` |
| `seed` | int | corpus seed |
| `records` | int | line count |
| `chars` | int | sum of `len(text)` |
| `tokens` | int | whitespace-split count (fixture). Unigram count happens at pack. |
| `sha256` | string | SHA-256 of the JSONL file bytes |
| `license` | string | `nullxes-synthetic-v0` |
| `generator` | string | generator id for that family |

Root `corpus.manifest.json` lists shards and a concatenated shard-hash digest. It is an index, not a substitute for per-shard manifests.

## Generators

- Seeded `random.Random` only. Same `(seed, per_layer)` ⇒ same JSONL bytes ⇒ same SHA.
- No LLM expander in v0.
- No chat templates, no `<|user|>` / `<|assistant|>` / `<|eot|>`, no `<|document|>`.
- No identity replay.
- Invented knowledge only: **Lerna-7**, **Varek**, **K-12** and names minted by the generator. No Wikipedia / FineWeb ingest.

Hard cap this cycle: 20_000 records total (`per_layer` × 5 layers × 2 languages). The 20–50M token build job is the next cycle after the contamination gate is green.

## Contamination gate (CPU)

A record is rejected if `text` contains:

- any `PRETRAIN_FORBIDDEN_STRINGS` or the substring `<|`
- identity needles (`i am shinra`, `я шинра`, …)
- instruction/chat wrappers (`### Instruction`, `User:`, `Assistant:`, `Вопрос:`, `Ответ:`)

Packed `input_ids` (later) must not contain EOT=2 or ids 4–17. Document wrap is `[BOS] + body + [END_OF_TEXT]`.

## Out of scope

G4, optimizer, FineWeb, Hub weight replace, tokenizer retrain, chat pretrain, LLM-v0 generation.
