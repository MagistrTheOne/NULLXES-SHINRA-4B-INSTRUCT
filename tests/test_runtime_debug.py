import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.debug import capture_runtime_snapshot


def test_capture_runtime_snapshot_cpu():
    snap = capture_runtime_snapshot("smoke", step=0, microbatch=0, sync_gradients=False)
    assert snap["tag"] == "smoke"
    assert snap["step"] == 0
    assert snap["microbatch"] == 0
    assert snap["sync_gradients"] is False
    assert "cuda_available" in snap
    assert "dtype" in snap
    assert "gpu_name" in snap
