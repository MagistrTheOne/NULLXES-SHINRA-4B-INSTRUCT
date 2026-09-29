"""SHINRA DATA V1: Phase A contract + Phase B local canary ingest.

No GPU. No training. Phase B does not fetch sources from the network.
Phase C (`phase_c.py`) packs FineWeb-Edu EN JSONL for an 8M honest-token
learning-pilot. GPU train is `scripts/v2_c_colab.py`. Hub weights and
`s1_language.yaml` stay closed.
Materialization ABI (`materialize.py`) registers an already-local JSONL slice
under scratch via receipt. Acquisition (`acquisition.py`) is a local-manifest contract. Gated FineWeb-Edu EN
fetch lives in `acquire_hf.py` (one revision, one parquet). Adapter and materializer
do not download. Canary uses scratch receipts. Git allowlist materialization stays unresolved.
`scripts/validate_data_v1_phase_a.py` is a second invocation path of the Phase A
contract, not an independent auditor.
"""


class PhaseBError(ValueError):
    """Canary ingest / gate failure."""
