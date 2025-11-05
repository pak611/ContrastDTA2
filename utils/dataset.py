from torch_geometric.data import InMemoryDataset, HeteroData, Data
import torch
from typing import List, Tuple, Dict, Any

class PLDataset(InMemoryDataset):
    """
    rows: list of (P:Data, L:Data, y:float, meta:dict)
    Keeps meta in a parallel list (not inside HeteroData) to avoid collation issues.
    """
    def __init__(self, rows, transform=None, pre_transform=None):
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)
        self.meta = []  # aligned with self.slices indexing

        data_list = []
        for P, L, y, meta in rows:
            d = HeteroData()
            # pocket store
            # x: [Np, Fp]
            d["pocket"].x = P.x if P.x.ndim == 2 else P.x.view(1, -1)
            # pos
            if hasattr(P, "pos"):
                d["pocket"].pos = P.pos
            # edges live in edge store ('pocket','pocket')
            d[("pocket", "pocket")].edge_index = P.edge_index  # [2, Ep]
            if hasattr(P, "edge_attr") and P.edge_attr is not None:
                d[("pocket", "pocket")].edge_attr = P.edge_attr  # [Ep, K]

            # ligand store
            d["ligand"].x = L.x if L.x.ndim == 2 else L.x.view(1, -1)
            # pos
            if hasattr(L, "pos"):
                d["ligand"].pos = L.pos
            # edges live in edge store ('ligand','ligand')
            d[("ligand", "ligand")].edge_index = L.edge_index  # [2, El]
            if hasattr(L, "edge_attr") and L.edge_attr is not None:
                d[("ligand", "ligand")].edge_attr = L.edge_attr  # [El, K]

            # target
            d.y = torch.tensor([float(y)], dtype=torch.float32)

            # (no target_id field in this reverted version)

            if pre_transform is not None:
                d = pre_transform(d)

            data_list.append(d)
            self.meta.append(meta)  # e.g., {'split':'train', 'id':'id_3zzf'}

        self.data, self.slices = self.collate(data_list)

    def get_meta(self, idx):
        return self.meta[idx]


class PLDataset2(InMemoryDataset):
    """
    Returns a HeteroData per sample with two graph stores:
      - data["protein"]: residue graph (x, edge_index[, edge_attr][, pos])
      - data["ligand"] : ligand graph  (x, edge_index[, edge_attr][, pos])
    Also attaches:
      - data.y : [1] float label

    rows: list of tuples (P:Data, L:Data, y:float, meta:dict)
    Meta is stored externally in self.meta to avoid collation issues.
    """
    def __init__(self,
                 rows: List[Tuple[Data, Data, float, Dict[str, Any]]],
                 transform=None,
                 pre_transform=None):
        super().__init__(root=None, transform=transform, pre_transform=pre_transform)
        self.meta = []

        data_list = []
        for P, L, y, meta in rows:
            d = HeteroData()

            # ---- protein graph store ----
            d["protein"].x = P.x
            # edges must live in EDGE STORE for HeteroData
            d[("protein", "protein")].edge_index = P.edge_index
            if hasattr(P, "edge_attr") and P.edge_attr is not None:
                d[("protein", "protein")].edge_attr = P.edge_attr
            if hasattr(P, "pos") and P.pos is not None:
                d["protein"].pos = P.pos
            # pass through handy ids if present
            for attr in ("protein_uid", "residue_keys"):
                if hasattr(P, attr):
                    setattr(d, attr, getattr(P, attr))

            # ---- ligand graph store ----
            d["ligand"].x = L.x
            d[("ligand", "ligand")].edge_index = L.edge_index
            if hasattr(L, "edge_attr") and L.edge_attr is not None:
                d[("ligand", "ligand")].edge_attr = L.edge_attr
            if hasattr(L, "pos") and L.pos is not None:
                d["ligand"].pos = L.pos
            for attr in ("ligand_id",):
                if hasattr(L, attr):
                    setattr(d, attr, getattr(L, attr))

            # ---- target ----
            d.y = torch.tensor([float(y)], dtype=torch.float32)

            # (no target_id field in this reverted version)

            if pre_transform is not None:
                d = pre_transform(d)

            data_list.append(d)
            self.meta.append(meta)

        self.data, self.slices = self.collate(data_list)

    def get_meta(self, idx: int) -> Dict[str, Any]:
        return self.meta[idx]