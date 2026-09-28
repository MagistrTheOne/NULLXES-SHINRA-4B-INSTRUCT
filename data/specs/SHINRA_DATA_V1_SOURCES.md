# SHINRA DATA V1 sources

**Status:** empty allowlist. Acquisition is **not** authorized in this commit.

Phase B here is the ingest **engine**: local JSONL/text, canonicalization, closed taxonomy, dedup, probe firewall, token cap, JSON report.

It does **not** download. It does **not** train. It does **not** read `configs/stages/s1_language.yaml`.

```text
allowed_sources: []
downloaders: none
network: forbidden
```

A later commit fills this file with licensed source ids, redistribution, and provenance snapshots. Until that allowlist exists, a real ≤20M canary ingest is closed.

Local fixtures for the engine live under `tests/fixtures/data_v1/phase_b/`.
