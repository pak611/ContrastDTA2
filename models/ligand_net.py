import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool

def mlp(cin, cout):
    return nn.Sequential(nn.Linear(cin, cout), nn.ReLU(), nn.Linear(cout, cout))

class LigandNet(nn.Module):
    def __init__(self, node_in, edge_in, hidden=256, layers=3, out_dim=256, dropout=0.1):
        super().__init__()
        self.node_proj = nn.Linear(node_in, hidden)
        self.edge_mlp  = nn.Sequential(nn.Linear(edge_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

        self.convs = nn.ModuleList([GINEConv(mlp(hidden, hidden)) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(2*hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, batch, return_nodes: bool = False):
        x, edge_index, edge_attr, b = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        h = self.node_proj(x)
        e = self.edge_mlp(edge_attr)  # expects edge_attr shape [E, edge_in]

        for conv, bn in zip(self.convs, self.norms):
            h = h + self.drop(torch.relu(bn(conv(h, edge_index, e))))

        if return_nodes:
            return h, b

        g = torch.cat([global_mean_pool(h, b), global_max_pool(h, b)], dim=-1)
        return self.head(g)  # [B, out_dim]
