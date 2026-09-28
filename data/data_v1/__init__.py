"""SHINRA DATA V1: Phase A contract + Phase B local canary ingest.

No GPU. No training. Phase B does not fetch sources from the network.
Materialization ABI (`materialize.py`) registers an already-local JSONL slice
under scratch via receipt. Acquisition (`acquisition.py`) is a local-manifest contract. Gated FineWeb-Edu EN
fetch lives in `acquire_hf.py` (one revision, one parquet). Adapter and materializer
do not download. Canary uses scratch receipts. Git allowlist materialization stays unresolved.
`scripts/validate_data_v1_phase_a.py` is a second invocation path of the Phase A
contract, not an independent auditor.
"""


class PhaseBError(ValueError):
    """Canary ingest / gate failure."""
