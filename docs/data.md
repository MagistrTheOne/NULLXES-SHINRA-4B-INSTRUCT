# SHINRA Pretraining Data Pipeline V2

See [`data_pipeline_v2.md`](data_pipeline_v2.md) for pinned sources, integrity contracts, Colab build/resume command, and admission conditions.

FineWeb-Edu and DCLM-Baseline-1.0 are configured as English candidates; FineWeb-2 `rus_Cyrl` is the Russian candidate. These exact-revision sources remain `audit_only`. The candidate source caps are not measured clean yields and do not establish legal permission to train. DCLM rights and PII review, source-level provenance, and near-dedup quality still require separate review.

The clean corpus is split by whole source work/document before tokenization and chunking. The Native SHINRA tokenizer is used without retraining. Packed training inventory must contain at least 500M actual supervised targets per language; the existing sampler handles 50/50 by presented target tokens and records any sequence reuse. Validation and blind stay in separate artifacts; training only opens train and validation.

No source is admitted to production automatically. `data.training_enabled`, `training.enabled` and `runtime.colab_launch_allowed` remain false until separate approval.
