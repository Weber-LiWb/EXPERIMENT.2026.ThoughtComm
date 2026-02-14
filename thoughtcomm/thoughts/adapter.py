from __future__ import annotations
import torch
import torch.nn as nn

class PrefixAdapter(nn.Module):
    """g: R^{nz} -> R^{m x d} (Eq. 11)."""
    def __init__(self, nz: int, d: int, m: int, hidden: int = 2048, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        in_dim = nz
        for _ in range(max(depth, 1)):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, m * d))
        self.net = nn.Sequential(*layers)
        self.m, self.d = m, d

    def forward(self, z_tilde: torch.Tensor) -> torch.Tensor:
        if z_tilde.dim() == 1:
            z_tilde = z_tilde.unsqueeze(0)
        out = self.net(z_tilde)
        return out.view(z_tilde.shape[0], self.m, self.d)
