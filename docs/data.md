# SHINRA Pretraining Data Pipeline V1

Source adapters, cleaning, document split assignment, lossless token chunking and packed artifacts are specified in [`data_pipeline_v1.md`](data_pipeline_v1.md). FineWeb, FineWeb-Edu, FineWeb-2 and SmolLM-Corpus remain excluded.

Wikimedia Wikipedia and PleIAs Russian-PD are pinned **candidates**. Their exact Hub commit IDs are recorded in `configs/shinra.yaml`; that pin freezes the bytes, not the legal status. Both candidate sources remain `audit_only` until separate rights and provenance reviews approve them. The configuration still keeps `data.training_enabled=false`, `training.enabled=false` and `runtime.colab_launch_allowed=false`.

The corpus cleaner preserves ordered numeric spans, source metadata and duplicate provenance, quarantines configured table/OCR patterns, uses a SQLite-backed exact and approximate MinHash index, and assigns deterministic splits by whole work/document before tokenization. Near-dedup recall/precision still needs review on the actual corpus before admission. `data.pack` creates independent, non-overlapping continuation windows. It never combines documents, drops source tokens or repeats source-token ranges. BOS is added and masked for every window; EOD is added and trained only on the final window of each source document.

No production corpus has been admitted or assembled by this implementation. No source download, corpus processing, packing, smoke test, or training was run while implementing the pipeline.
