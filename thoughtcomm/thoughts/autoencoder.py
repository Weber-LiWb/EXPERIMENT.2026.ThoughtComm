from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, depth: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(max(depth, 1)):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseJacobianAutoEncoder(nn.Module):
    """Sparsity-regularized autoencoder for extracting latent thoughts.

    Corresponds to Eq.(6)-(7): z = \hat f^{-1}(H), H_hat = \hat f(z),
    with a sparsity penalty on J_{\hat f} (decoder Jacobian).
    """

    def __init__(self, nh: int, nz: int, hidden: int = 2048, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        self.nh, self.nz = nh, nz
        self.enc = MLP(nh, nz, hidden=hidden, depth=depth, dropout=dropout)
        self.dec = MLP(nz, nh, hidden=hidden, depth=depth, dropout=dropout)

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return self.enc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(h)
        h_hat = self.decode(z)
        return z, h_hat


def jacobian_l1_penalty(
    ae: SparseJacobianAutoEncoder,
    z: torch.Tensor,
    mode: Literal["sampled_rows", "exact"] = "sampled_rows",
    sampled_rows_k: int = 256,
) -> torch.Tensor:
    """Estimate ||J_dec(z)||_1.

    - exact: full Jacobian rows (O(nh*nz), batch_size must be 1)
    - sampled_rows: sample k output dims per item and scale.
    """
    z = z.requires_grad_(True)
    h_hat = ae.decode(z)

    if mode == "exact":
        assert z.shape[0] == 1, "exact mode expects batch_size=1 (engineering choice)."
        nh = h_hat.shape[-1]
        pen = 0.0
        for q in range(nh):
            grad_q = torch.autograd.grad(h_hat[0, q], z, retain_graph=True, create_graph=True)[0]
            pen = pen + grad_q.abs().sum()
        return pen / nh

    B, nh = h_hat.shape
    k = min(sampled_rows_k, nh)
    idx = torch.randint(0, nh, (B, k), device=h_hat.device)

    pen = 0.0
    # Row-by-row preserves the L1-of-J definition; it is slower but faithful.
    for t in range(k):
        out_t = h_hat.gather(1, idx[:, t].unsqueeze(1)).sum()
        grad = torch.autograd.grad(out_t, z, retain_graph=True, create_graph=True)[0]
        pen = pen + grad.abs().mean()
    return pen * (nh / k)


@torch.no_grad()
def estimate_structure_mask(
    ae: SparseJacobianAutoEncoder,
    states: torch.Tensor,          # (N, nh) on CPU or GPU
    num_agents: int,
    hidden_size: int,
    threshold: float = 1e-4,
    rows_per_sample: int = 256,
    max_samples: int | None = None,
) -> torch.Tensor:
    """Estimate B(J_dec): (num_agents, nz) boolean mask.

    For each sample, we:
    - compute z = enc(h)
    - decode to h_hat
    - sample output rows q, compute |∂h_hat[q]/∂z|, aggregate to agent slice.
    """
    device = next(ae.parameters()).device
    ae.eval()
    states = states.to(device)

    N, nh = states.shape
    if max_samples is not None:
        N = min(N, int(max_samples))
        states = states[:N]

    nz = ae.nz
    acc = torch.zeros((num_agents, nz), device=device)
    cnt = torch.zeros((num_agents, 1), device=device)

    for n in range(N):
        h = states[n:n+1]
        z = ae.encode(h).requires_grad_(True)
        h_hat = ae.decode(z)

        k = min(rows_per_sample, nh)
        rows = torch.randint(0, nh, (k,), device=device)
        for q in rows:
            out_q = h_hat[0, q]
            grad = torch.autograd.grad(out_q, z, retain_graph=True, create_graph=False)[0]
            agent_i = int(q.item() // hidden_size)
            acc[agent_i] += grad.abs().squeeze(0)
            cnt[agent_i] += 1.0

    mean_grad = acc / cnt.clamp_min(1.0)
    mask = mean_grad > threshold
    return mask.detach().cpu()
