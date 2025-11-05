import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool

class RBF(nn.Module):
    def __init__(self, K=16, dmin=0.0, dmax=12.0):
        super().__init__()
        centers = torch.linspace(dmin, dmax, K)          # [K]
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / ((dmax - dmin) / K)**2
    def forward(self, d):                                 # d: [E,1] distances in Å
        return torch.exp(-self.gamma * (d - self.centers)**2)  # [E,K]

def _node_mlp(cin, cout):
    return nn.Sequential(nn.Linear(cin, cout), nn.ReLU(), nn.Linear(cout, cout))

class PocketEncoder(nn.Module):
    def __init__(self, node_in=20, hidden=128, layers=3, rbf_K=16):
        super().__init__()
        self.node_proj = nn.Linear(node_in, hidden)
        self.rbf = RBF(rbf_K)
        self.edge_mlp = nn.Sequential(nn.Linear(rbf_K, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

        self.convs = nn.ModuleList([GINEConv(_node_mlp(hidden, hidden)) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dropout = nn.Dropout(0.1)

        self.readout = nn.Sequential(
            nn.Linear(2*hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

    def forward(self, data, return_nodes: bool = False):
        # data: PyG Data with x (N,20), edge_index, edge_attr (E,1), and batch
        h = self.node_proj(data.x)                        # [N,H]
        e = self.edge_mlp(self.rbf(data.edge_attr))       # [E,H]

        for conv, ln in zip(self.convs, self.norms):
            h = h + self.dropout(torch.relu(ln(conv(h, data.edge_index, e))))

        if return_nodes:
            return h, data.batch                          # node embeddings + batch

        g = torch.cat([global_mean_pool(h, data.batch),
                       global_max_pool(h, data.batch)], dim=-1)   # [B,2H]
        return self.readout(g)                                    # [B,H]
