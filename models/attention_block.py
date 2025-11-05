import math
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool

class CrossAttentionBlock(nn.Module):
    """
    Bi-directional multihead cross-attention over pocket & ligand nodes per graph.
    Inputs:
      Hp_nodes: [Np_total, D], batch_p: [Np_total]
      Hl_nodes: [Nl_total, D], batch_l: [Nl_total]
    Output:
      Hf: [B, D] fused per-graph vector
    """

    def __init__(self, d, heads=4, dropout=0.1, map_dim: int | None = None):
        super().__init__()
        self.d = d
        self.heads = heads
        self.att_pl = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)  # pocket queries ligand
        self.att_lp = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)  # ligand queries pocket
        # Projections for interaction map (per-head)
        if map_dim is None:
            map_dim = max(8, d // heads)
        self.map_dim = map_dim
        self.scale = math.sqrt(self.map_dim)
        self.d_a = nn.Linear(d, heads * self.map_dim)  # ligand/drug projection
        self.p_a = nn.Linear(d, heads * self.map_dim)  # protein/pocket projection
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        # Input to FF is concatenation of pooled stats from both sets: gp [4D] + gl [4D] => 8D
        self.ff = nn.Sequential(
            nn.Linear(8 * d, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

    @staticmethod
    def _split_by_batch(H, batch):
        """Return a list of [Ni, D] tensors per graph id."""
        B = int(batch.max().item()) + 1
        return [H[batch == i] for i in range(B)]

    def forward(self, Hp_nodes, batch_p, Hl_nodes, batch_l, return_map: bool = False):
        P_list = self._split_by_batch(Hp_nodes, batch_p)
        L_list = self._split_by_batch(Hl_nodes, batch_l)

        out_vecs = []
        maps = [] if return_map else None
        for P, L in zip(P_list, L_list):
            # shapes: P:[Np,D], L:[Nl,D]
            # pocket→ligand (queries=P, keys/values=L)
            Cp, _ = self.att_pl(query=P.unsqueeze(0), key=L.unsqueeze(0), value=L.unsqueeze(0))
            # ligand→pocket
            Cl, _ = self.att_lp(query=L.unsqueeze(0), key=P.unsqueeze(0), value=P.unsqueeze(0))

            # pool updated node sets
            gp = torch.cat([P.mean(0),    P.max(0).values,
                            Cp.squeeze(0).mean(0), Cp.squeeze(0).max(0).values], dim=-1)  # [4D]
            gl = torch.cat([L.mean(0),    L.max(0).values,
                            Cl.squeeze(0).mean(0), Cl.squeeze(0).max(0).values], dim=-1)  # [4D]

            # compress to D with small FF
            out_vecs.append(self.ff(torch.cat([gp, gl], dim=-1).view(1, -1)))  # [1, D]

            if return_map:
                # Build per-graph interaction map across heads
                Nl = L.size(0)
                Np = P.size(0)
                if Nl == 0 or Np == 0:
                    maps.append(torch.zeros(Nl, Np, device=L.device))
                else:
                    # L:[Nl,D] -> [H, Nl, d_ef]
                    d_proj = self.relu(self.d_a(L)).view(Nl, self.heads, self.map_dim).permute(1, 0, 2)
                    # P:[Np,D] -> [H, Np, d_ef]
                    p_proj = self.relu(self.p_a(P)).view(Np, self.heads, self.map_dim).permute(1, 0, 2)
                    # [H, Nl, d_ef] @ [H, d_ef, Np] -> [H, Nl, Np]
                    inter = torch.matmul(d_proj, p_proj.transpose(1, 2)) / self.scale
                    inter = self.tanh(inter).mean(dim=0)  # [Nl, Np] (average over heads)
                    maps.append(inter)
        Hf = torch.cat(out_vecs, dim=0)  # [B, D]
        return (Hf, maps) if return_map else Hf
