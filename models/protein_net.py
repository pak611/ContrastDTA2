import torch
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool

class _RBF(nn.Module):
    def __init__(self, K=16, dmin=0.0, dmax=12.0):
        super().__init__()
        centers = torch.linspace(dmin, dmax, K)
        self.register_buffer("centers", centers)
        # width so adjacent centers overlap reasonably
        self.gamma = 1.0 / ((dmax - dmin) / K)**2
    def forward(self, d):                      # d: [E,1] (Å)
        return torch.exp(-self.gamma * (d - self.centers)**2)  # [E,K]

def _mlp(cin, cout):
    return nn.Sequential(nn.Linear(cin, cout), nn.ReLU(), nn.Linear(cout, cout))

class ProteinNet(nn.Module):
    """
    Residue-graph encoder:
      - node_in: residue feature dim (e.g., 20 AA one-hot [+ scalars] → 21)
      - edge_attr: distances in Å (will be RBF-expanded internally)
    Returns a graph embedding of size out_dim.
    """
    def __init__(self,
                 node_in: int = 21,
                 hidden: int = 256,
                 layers: int = 3,
                 out_dim: int = 256,
                 rbf_K: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        self.node_proj = nn.Linear(node_in, hidden)
        self.rbf = _RBF(K=rbf_K, dmin=0.0, dmax=12.0)
        self.edge_mlp = nn.Sequential(nn.Linear(rbf_K, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

        self.convs = nn.ModuleList([GINEConv(_mlp(hidden, hidden)) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)

        # readout + projection to out_dim
        self.readout = nn.Sequential(
            nn.Linear(2*hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def _ensure_edge_attr(self, x, edge_index, edge_attr, pos):
        """
        If edge_attr is None but pos is available, compute distances.
        Otherwise fall back to zeros (model will rely on topology only).
        """
        if edge_attr is not None:
            return edge_attr
        if pos is not None and edge_index.numel() > 0:
            # compute Euclidean distance per edge
            src, dst = edge_index[0], edge_index[1]
            d = (pos[src] - pos[dst]).pow(2).sum(-1).sqrt().view(-1, 1)  # [E,1]
            return d
        # fallback: zeros → RBF will be peaked at dmin
        return torch.zeros(edge_index.size(1), 1, device=x.device, dtype=x.dtype)

    def forward(self, data, return_nodes: bool = False):
        """
        data: either a homogeneous protein Data, or a HeteroData containing
              node store ["protein"] and edge store ("protein","protein").
        If return_nodes=True, returns (h, batch) with node embeddings and batch vector.
        Otherwise returns pooled graph embeddings [B, out_dim].
        """
        if isinstance(data, HeteroData) or hasattr(data, "node_types"):
            nstore = data["protein"]
            estore = data[("protein", "protein")]
            x = nstore.x
            edge_index = estore.edge_index
            edge_attr = getattr(estore, "edge_attr", None)
            pos = getattr(nstore, "pos", None)
            batch = getattr(nstore, "batch", None)
        else:
            x = data.x
            edge_index = data.edge_index
            edge_attr = getattr(data, "edge_attr", None)
            pos = getattr(data, "pos", None)
            batch = getattr(data, "batch", None)
        if batch is None:
            # if not batched yet, create a dummy batch of zeros
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        # Node/edge projections
        h = self.node_proj(x)                                    # [N,H]
        d = self._ensure_edge_attr(x, edge_index, edge_attr, pos)  # [E,1]
        e = self.edge_mlp(self.rbf(d))                           # [E,H]

        # Message passing (residual + LN + Dropout)
        for conv, ln in zip(self.convs, self.norms):
            h = h + self.drop(torch.relu(ln(conv(h, edge_index, e))))

        if return_nodes:
            return h, batch

        # Readout: mean+max pooling
        g = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1)  # [B, 2H]
        z = self.readout(g)  # [B, out_dim]
        return z
