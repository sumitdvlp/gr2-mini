"""Device / dtype helpers that encode the quirks of a GTX 1660 Ti.  (see README §3)"""
from __future__ import annotations

import torch


def get_device(prefer: str = "cuda") -> torch.device:
    """Pick a device.

    prefer = "cuda" or "auto" -> cuda if torch.cuda.is_available() else cpu
    prefer = "cpu"            -> cpu
    anything else             -> ValueError
    """
    if prefer in ("cuda", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if prefer == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Invalid device preference: {prefer!r}")


def pick_dtype(device: torch.device, prefer: str = "fp16") -> torch.dtype:
    """Choose a compute dtype for `device`.

    * device.type == "cpu"           -> torch.float32, whatever `prefer` says
    * cuda, prefer == "fp32"         -> torch.float32
    * cuda, prefer == "fp16"         -> torch.float16
    * cuda, prefer == "bf16"         -> torch.bfloat16 only if
                                        torch.cuda.get_device_capability(device) >= (8, 0),
                                        otherwise torch.float16
    * any other `prefer`             -> ValueError

    Do NOT use torch.cuda.is_bf16_supported(): it also counts slow emulated bf16.
    """
    if device.type == "cpu":
        if prefer not in ("fp16", "bf16", "fp32"):
            raise ValueError(f"Invalid prefer={prefer!r} for device {device}")
        return torch.float32
    if prefer == "fp32":
        return torch.float32
    if prefer == "fp16":
        return torch.float16
    if prefer == "bf16":
        cc = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if cc >= (8, 0) else torch.float16
    raise ValueError(f"Invalid prefer={prefer!r} for device {device}")


def vram_info(device: torch.device) -> dict | None:
    """Return GPU memory info, or None for a CPU device.

    For CUDA: {"name": str, "total_gb": float, "free_gb": float}
    using torch.cuda.get_device_name and torch.cuda.mem_get_info (bytes -> GiB, i.e. / 1024**3).
    """
    if device.type != "cuda":
        return None
    name = torch.cuda.get_device_name(device)
    free, total = torch.cuda.mem_get_info(device)
    return {"name": name, "total_gb": total / 1024**3, "free_gb": free / 1024**3}


def fp16_sanity_check(device: torch.device, n: int = 256, seed: int = 0) -> dict:
    """Compare an fp16 matmul against fp32 on `device`.

    Steps:
        1. Make two random [n, n] fp32 matrices A, B on `device` (use a seeded torch.Generator
           on the CPU, then move them, so the result doesn't depend on global RNG state).
        2. ref = A @ B (fp32);  half = (A.half() @ B.half()).float()
        3. finite      = bool(torch.isfinite(half).all())
           max_rel_err = float(max |half - ref| / max |ref|)
           ok          = finite and max_rel_err < 1e-2

    Returns {"ok": bool, "finite": bool, "max_rel_err": float}.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.rand(n, n, dtype=torch.float32, generator=g, device=device)
    B = torch.rand(n, n, dtype=torch.float32, generator=g, device=device)
    ref = A @ B
    half = (A.half() @ B.half()).float()
    finite = bool(torch.isfinite(half).all())
    max_rel_err = float((half - ref).abs().max() / ref.abs().max())
    ok = finite and max_rel_err < 1e-2
    return {"ok": ok, "finite": finite, "max_rel_err": max_rel_err}
