"""FSDP wrapping for SHINRA on A100 nodes."""

from __future__ import annotations

from functools import partial

import torch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from model.modeling_shinra import ShinraDecoderLayer


def shinra_auto_wrap_policy():
    return partial(transformer_auto_wrap_policy, transformer_layer_cls={ShinraDecoderLayer})


def enable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
