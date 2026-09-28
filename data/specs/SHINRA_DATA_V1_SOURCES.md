# SHINRA DATA V1 sources

**Status:** two sources **approved for governance only**. Acquisition **closed**. Real canary **closed**.

```text
downloaders: none
network: forbidden
acquisition: closed
materialization: not_materialized
real canary: fail-closed until slice_id + content_sha256 exist
```

Machine copy: [`sources.allowlist.json`](sources.allowlist.json)  
Schema: [`sources.schema.json`](sources.schema.json)

`configs/stages/s1_language.yaml` is not a source. It stays unauthorized synth-v0 until Phase D.

**Allowlist approval ≠ acquisition authorization.**  
**Allowlist approval ≠ real-canary authorization.**

---

## Approved source identities

Independently authored web text. Not a translation pair. Not chat. Not synth-v0. Not a probe set.

| source_id | upstream | subset | language |
|---|---|---|---|
| `fineweb-edu-en` | `HuggingFaceFW/fineweb-edu` | unresolved (`null`) | `en` |
| `fineweb2-ru` | `HuggingFaceFW/fineweb-2` | **`rus_Cyrl`** (required) | `ru` |

Official cards (evidence, not acquisition URLs in the allowlist JSON):

- FineWeb-Edu: Hugging Face dataset `HuggingFaceFW/fineweb-edu`, Hub license `odc-by`, card text **ODC-By v1.0**, plus Common Crawl Terms of Use.
- FineWeb2: Hugging Face dataset `HuggingFaceFW/fineweb-2`, Hub license `odc-by`, card text **ODC-By v1.0**, plus Common Crawl Terms of Use. Russian split is `rus_Cyrl`.

`upstream.revision` is **null**. No commit/tag was verified as a freeze; do not invent one.

---

## License vs crawled pages

`license.dataset_redistribution: true` means the **dataset compilation** is under ODC-By-1.0 (attribution). It does **not** mean crawled page bodies are free of third-party rights.

`underlying_content_caveat: common_crawl_third_party_rights` and `common_crawl_tou: true` are mandatory for these two records. Common Crawl warns that users must consider rights in the underlying crawled content.

Do not collapse that into a single boolean.

---

## Slices, not dumps

FineWeb-Edu is terabyte-scale. FineWeb2 `rus_Cyrl` is documented at **1.81 TB disk / 5.82 TB UTF-8**. Full materialization is **forbidden**.

Per source: `max_materialized_bytes = 4294967296` (4 GiB).  
Global: `disk.max_materialized_bytes = 8589934592` (8 GiB).  
Per source canary tokens: max **8,000,000**. Global canary tokens: **20,000,000**.

Materialize a **bounded local jsonl slice** outside this engine. Sidecar must not store raw URLs (FineWeb rows have a `url` field; hash it, do not copy it).

---

## Token accounting

| run | counter | rule |
|---|---|---|
| fixture dry-run | whitespace **allowed** | engine tests only |
| real corpus canary | **tokenizer required** | `encode(text, add_special_tokens=False)` |
| canary cap | 20_000_000 | tokenizer tokens of the **body** |

`--mode canary` fails while `materialization.status != materialized` or `content_sha256` is null.

---

## Disk (live Colab, not YAML)

Observed G4 session: **235.7 GB total / 106.9 GB used / ~128.8 GB free**.  
`configs/colab.yaml` `disk_ceiling_gb: 400` is **not** physical capacity.

```text
scratch:     /content/shinra_scratch/data_v1
max raw:     8589934592 bytes (8 GiB)
min free:    20 GB
S0 scratch:  /content/shinra_scratch/s0  — do not delete
Drive:       forbidden on the hot path
```

---

## Opening a real canary (later)

1. Materialize bounded slices locally (outside this repo engine).
2. Fill `materialization.slice_id` and `content_sha256`.
3. Keep `acquisition: closed` until that fill is reviewed.
4. `python -m data.data_v1.phase_b --mode canary --tokenizer tokenizer/artifacts/tokenizer.json --input <local jsonl>`
5. No training. No S1. No downloaders in `data/data_v1`.
