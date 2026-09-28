"""SHINRA DATA V1: Phase A contract + Phase B local canary ingest.

No GPU. No training. Phase B does not fetch sources from the network.
`scripts/validate_data_v1_phase_a.py` is a second invocation path of the Phase A
contract, not an independent auditor.
"""


class PhaseBError(ValueError):
    """Canary ingest / gate failure."""
