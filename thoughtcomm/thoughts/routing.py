from __future__ import annotations
from typing import Dict, Optional
import torch

def agreement_alpha(mask: torch.Tensor) -> torch.Tensor:
    """mask: (A, nz) bool -> alpha: (nz,) counts in [0..A]"""
    return mask.to(torch.int32).sum(dim=0)

def build_weight_vector(alpha: torch.Tensor, mode: str = "linear", custom: Optional[Dict[int, float]] = None) -> torch.Tensor:
    device = alpha.device
    if mode == "uniform":
        w = torch.ones_like(alpha, dtype=torch.float32, device=device)
        return torch.where(alpha == 0, torch.zeros_like(w), w)

    if mode == "linear":
        max_a = int(alpha.max().item()) if alpha.numel() else 1
        w = alpha.to(torch.float32) / max(1, max_a)
        return w

    if mode == "custom":
        custom = custom or {}
        w = torch.ones_like(alpha, dtype=torch.float32, device=device)
        for a, val in custom.items():
            w = torch.where(alpha == int(a), torch.tensor(float(val), device=device), w)
        w = torch.where(alpha == 0, torch.zeros_like(w), w)
        return w

    raise ValueError(f"Unknown mode: {mode}")

def route_latents(z: torch.Tensor, mask: torch.Tensor, weight_vec: torch.Tensor) -> torch.Tensor:
    """z: (nz,) or (B,nz) -> z_tilde: (A,nz) or (B,A,nz)."""
    if z.dim() == 1:
        z_ = z.unsqueeze(0)  # (1,nz)
        z_t = z_.repeat(mask.shape[0], 1) * mask.to(z_.dtype)
        return z_t * weight_vec.unsqueeze(0)

    # batch: (B,nz)
    B = z.shape[0]
    A = mask.shape[0]
    z_t = z.unsqueeze(1).repeat(1, A, 1) * mask.to(z.dtype).unsqueeze(0)
    return z_t * weight_vec.view(1, 1, -1)
