from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import torch


def _cuda_memory_snapshot() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "memory_allocated": 0,
        "memory_reserved": 0,
        "max_memory_allocated": 0,
        "max_memory_reserved": 0,
        "free_vram": None,
        "total_vram": None,
    }
    if not torch.cuda.is_available():
        return payload
    try:
        device = torch.cuda.current_device()
        payload["gpu_name"] = torch.cuda.get_device_name(device)
        payload["memory_allocated"] = int(torch.cuda.memory_allocated(device))
        payload["memory_reserved"] = int(torch.cuda.memory_reserved(device))
        payload["max_memory_allocated"] = int(torch.cuda.max_memory_allocated(device))
        payload["max_memory_reserved"] = int(torch.cuda.max_memory_reserved(device))
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        payload["free_vram"] = int(free_bytes)
        payload["total_vram"] = int(total_bytes)
    except Exception:
        pass
    return payload


def capture_runtime_snapshot(
    tag: str,
    *,
    step: int = 0,
    microbatch: int = 0,
    sync_gradients: bool = False,
    model: torch.nn.Module | None = None,
    dtype: torch.dtype | None = None,
    reset_peak: bool = False,
) -> dict[str, Any]:
    cuda = _cuda_memory_snapshot()
    if reset_peak and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
            cuda = _cuda_memory_snapshot()
        except Exception:
            pass
    if model is not None and dtype is None:
        try:
            params = list(model.parameters())
            if params:
                dtype = params[0].dtype
        except Exception:
            dtype = None
    if dtype is None:
        dtype = torch.get_default_dtype()
    payload = {
        "tag": tag,
        "step": int(step),
        "microbatch": int(microbatch),
        "sync_gradients": bool(sync_gradients),
        "cuda_available": cuda["cuda_available"],
        "gpu_name": cuda["gpu_name"],
        "dtype": str(dtype),
        "memory_allocated": cuda["memory_allocated"],
        "memory_reserved": cuda["memory_reserved"],
        "max_memory_allocated": cuda["max_memory_allocated"],
        "max_memory_reserved": cuda["max_memory_reserved"],
        "free_vram": cuda["free_vram"],
        "total_vram": cuda["total_vram"],
    }
    return payload


def summarize_gradients(model: torch.nn.Module) -> dict[str, Any]:
    dtype_counts: Counter[str] = Counter()
    total_elements = 0
    total_bytes = 0
    nan_count = 0
    inf_count = 0
    non_none = 0
    abs_max_values: list[float] = []
    grad_norm_terms: list[torch.Tensor] = []
    for param in model.parameters():
        if param.grad is None:
            continue
        non_none += 1
        grad = param.grad.detach()
        dtype_counts[str(grad.dtype)] += 1
        total_elements += int(grad.numel())
        total_bytes += int(grad.numel() * grad.element_size())
        nan_count += int(torch.isnan(grad).sum().item())
        inf_count += int(torch.isinf(grad).sum().item())
        if grad.numel() > 0:
            abs_max_values.append(float(grad.abs().max().item()))
            grad_norm_terms.append(grad.float().reshape(-1).norm(p=2))
    grad_norm = None
    if grad_norm_terms:
        grad_norm = float(torch.stack(grad_norm_terms).sum().item())
    return {
        "parameters_with_grad": non_none,
        "grad_dtype_distribution": dict(dtype_counts),
        "total_gradient_elements": total_elements,
        "nan_gradients": nan_count,
        "inf_gradients": inf_count,
        "global_grad_norm": grad_norm,
        "min_abs_grad": min(abs_max_values) if abs_max_values else None,
        "max_abs_grad": max(abs_max_values) if abs_max_values else None,
        "approx_grad_bytes": total_bytes,
    }


def summarize_optimizer_state(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    dtype_counts: Counter[str] = Counter()
    dtype_elements: Counter[str] = Counter()
    dtype_bytes: Counter[str] = Counter()
    device_counts: Counter[str] = Counter()
    tensor_count = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state[p]
            if not state:
                continue
            for value in state.values():
                if not isinstance(value, torch.Tensor):
                    continue
                tensor_count += 1
                dtype_counts[str(value.dtype)] += 1
                dtype_elements[str(value.dtype)] += int(value.numel())
                dtype_bytes[str(value.dtype)] += int(value.numel() * value.element_size())
                device_counts[str(value.device)] += 1
    return {
        "state_tensor_count": tensor_count,
        "dtype_distribution": dict(dtype_counts),
        "elements_by_dtype": dict(dtype_elements),
        "bytes_by_dtype": dict(dtype_bytes),
        "device_distribution": dict(device_counts),
        "estimated_optimizer_state_bytes": sum(dtype_bytes.values()),
    }


def summarize_attention_tensor(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": tuple(tensor.stride()),
        "is_contiguous": bool(tensor.is_contiguous()),
    }
