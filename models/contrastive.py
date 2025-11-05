from __future__ import annotations

import torch
import torch.nn as nn


class SymmetricInfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.07, learnable: bool = True):
        super().__init__()
        if learnable:
            self.log_tau = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer("log_tau", torch.log(torch.tensor(temperature)), persistent=False)

    @property
    def tau(self) -> torch.Tensor:
        # clamp temperature for stability
        return self.log_tau.exp().clamp(1e-3, 1.0)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute symmetric InfoNCE loss for two sets of normalized embeddings.

        z1, z2: [B, D], assumed L2-normalized.
        """
        z1 = torch.nn.functional.normalize(z1, dim=-1)
        z2 = torch.nn.functional.normalize(z2, dim=-1)
        sim = z1 @ z2.t()  # [B, B]
        sim = sim / self.tau
        target = torch.arange(sim.size(0), device=sim.device)
        loss1 = torch.nn.functional.cross_entropy(sim, target)
        loss2 = torch.nn.functional.cross_entropy(sim.t(), target)
        return 0.5 * (loss1 + loss2)


__all__ = ["SymmetricInfoNCE"]
