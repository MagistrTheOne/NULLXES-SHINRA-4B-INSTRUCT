"""CUDA math settings for the current single-process trainer; FSDP is not implemented."""

from __future__ import annotations

import torch


def enable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
