import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from models.attention_block import CrossAttentionBlock
from models.protein_net import ProteinNet

class BaseDTA(nn.Module):
    """
    Two-branch model: pocket_encoder and ligand_encoder with cross-attention fusion.
    Expects HeteroData with:
      data["pocket"]: x, edge_index, edge_attr, batch
      data["ligand"]: x, edge_index, (edge_attr), batch
    Protein encoding is intentionally omitted in this variant.
    """
    def __init__(self, pocket_encoder, ligand_encoder, d_latent=256, n_heads=4, dropout=0.1):
        super().__init__()
        self.pocket = pocket_encoder      # returns [B, d_latent]
        self.ligand = ligand_encoder      # returns [B, d_latent]
        # Cross-attention fusion over node embeddings
        self.fuse_pl = CrossAttentionBlock(d=d_latent, heads=n_heads, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(d_latent, d_latent), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_latent, 1)  # continuous outcome
        )

    def _to_homo(self, hd, ntype: str) -> Data:
        # Build a homogeneous Data view for a node type using its self-edge store
        nstore = hd[ntype]
        estore = hd[(ntype, ntype)]
        # edge_index may be missing; default to empty
        edge_index = getattr(estore, 'edge_index', None)
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=nstore.x.device)
        g = Data(x=nstore.x, edge_index=edge_index, batch=nstore.batch)
        if hasattr(nstore, "pos"):
            g.pos = nstore.pos
        # Ensure edge_attr is a tensor (fallback to ones if missing/None)
        eattr = getattr(estore, 'edge_attr', None)
        if eattr is None:
            E = g.edge_index.size(1)
            g.edge_attr = torch.ones((E, 1), dtype=nstore.x.dtype, device=nstore.x.device)
        else:
            g.edge_attr = eattr
        return g

    def forward(self, data, return_maps: bool = False):
        # 1) encode pocket & ligand to node-level embeddings
        p_h, p_b = self.pocket(self._to_homo(data, "pocket"), return_nodes=True)   # [Np,H], [Np]
        l_h, l_b = self.ligand(self._to_homo(data, "ligand"), return_nodes=True)   # [Nl,H], [Nl]
        # 2) fuse pocket↔ligand via cross-attention into per-graph vector
        if return_maps:
            H, maps = self.fuse_pl(p_h, p_b, l_h, l_b, return_map=True)  # [B,H], List[[Nl_i,Np_i]]
        else:
            H = self.fuse_pl(p_h, p_b, l_h, l_b)                 # [B, H]
        # 3) predict
        yhat = self.head(H).squeeze(-1)        # [B]
        if return_maps:
            return {"pred": yhat, "maps": maps}
        return yhat


class BaseDTA2(nn.Module):
    """
    Two-branch model: protein and ligand encoders with cross-attention fusion (like BaseDTA).
    Expects HeteroData with:
      data["protein"]: x, edge_index, (edge_attr), batch
      data["ligand"]:  x, edge_index, (edge_attr), batch
    """
    def __init__(self, protein_encoder: ProteinNet, ligand_encoder, d_latent=256, n_heads=4, dropout=0.1):
        super().__init__()
        self.protein = protein_encoder    # returns node embeddings with return_nodes=True
        self.ligand = ligand_encoder      # returns node embeddings with return_nodes=True
        self.fuse_pl = CrossAttentionBlock(d=d_latent, heads=n_heads, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(d_latent, d_latent), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_latent, 1)
        )

    def _to_homo(self, hd, ntype: str) -> Data:
        nstore = hd[ntype]
        estore = hd.get((ntype, ntype), None)
        g = Data(x=nstore.x)
        if hasattr(nstore, "batch"):
            g.batch = nstore.batch
        if estore is not None:
            edge_index = getattr(estore, 'edge_index', None)
            if edge_index is None:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=nstore.x.device)
            g.edge_index = edge_index
            eattr = getattr(estore, 'edge_attr', None)
            if eattr is None:
                E = g.edge_index.size(1)
                g.edge_attr = torch.ones((E, 1), dtype=nstore.x.dtype, device=nstore.x.device)
            else:
                g.edge_attr = eattr
        else:
            # no edge store, create empty edges and default attr
            g.edge_index = torch.empty((2, 0), dtype=torch.long, device=nstore.x.device)
            g.edge_attr = torch.ones((0, 1), dtype=nstore.x.dtype, device=nstore.x.device)
        return g

    def forward(self, data, return_maps: bool = False):
        # Protein: node embeddings + batch
        p_h, p_b = self.protein(data, return_nodes=True)
        # Ligand: node embeddings + batch
        l_h, l_b = self.ligand(self._to_homo(data, "ligand"), return_nodes=True)
        # Cross-attention fusion
        if return_maps:
            H, maps = self.fuse_pl(p_h, p_b, l_h, l_b, return_map=True)
        else:
            H = self.fuse_pl(p_h, p_b, l_h, l_b)  # [B, d_latent]
        yhat = self.head(H).squeeze(-1)
        if return_maps:
            return {"pred": yhat, "maps": maps}
        return yhat
