from __future__ import annotations

from typing import List, Dict, Union, Tuple, Optional

import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType as CT, HybridizationType as HT
import torch
import esm  # Facebook AI Research ESM models
from torch_geometric.data import Data, Batch
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


import re, numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
import torch
from torch_geometric.data import Data

import re, numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
import torch
from torch_geometric.data import Data
from pathlib import Path

BACKBONE = {"N","CA","C","O"}
AA_LIST = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
           'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
AA_TO_IDX = {aa:i for i,aa in enumerate(AA_LIST)}

def parse_fpocket_pdb(path):
    """
    Parse an fpocket pocket-atoms PDB (contacted atoms only).
    Returns: dict[(chain, resseq, resname)] -> list of atom dicts {name, elem, xyz}
    """
    residues = defaultdict(list)

    with open(path, 'r') as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            atom_name = line[12:16].strip()
            resname   = line[17:20].strip()
            chain_id  = line[21].strip() if len(line) > 21 else ''
            try:
                resseq = int(line[22:26])
            except ValueError:
                m = re.search(r'\s([A-Za-z])\s+(-?\d+)\s', line)
                resseq = int(m.group(2)) if m else 0

            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except ValueError:
                nums = re.findall(r'(-?\d+\.\d+)', line)
                if len(nums) >= 3:
                    x, y, z = map(float, nums[:3])
                else:
                    continue

            elem = (line[76:78].strip() if len(line) >= 78 else atom_name[0]).upper()
            if elem == 'H':  # skip hydrogens
                continue

            key = (chain_id, resseq, resname.upper())
            residues[key].append({
                "name": atom_name,
                "elem": elem,
                "xyz": np.array([x, y, z], dtype=np.float32)
            })

    return residues

def residue_centroid_fpocket(atom_list):
    """Return centroid of side-chain atoms; fallback to all heavy atoms."""
    sc = [a["xyz"] for a in atom_list if a["name"] not in BACKBONE]
    if len(sc) == 0:
        sc = [a["xyz"] for a in atom_list]
    if len(sc) == 0:
        return None
    return np.vstack(sc).mean(axis=0).astype(np.float32)

def build_pocket_graph_from_fpocket(path, k=16, use_radius_edges=False, radius=8.0):
    """
    Build a PyTorch Geometric Data object from an fpocket pocket-atoms PDB.
    Node = residue centroid (side-chain if available).
    Edge = kNN or radius-based spatial proximity.
    """
    resmap = parse_fpocket_pdb(path)

    node_xyz, node_feats, residue_keys = [], [], []
    for (chain, resseq, resname), atoms in resmap.items():
        xyz = residue_centroid_fpocket(atoms)
        if xyz is None:
            continue

        node_xyz.append(xyz)
        onehot = np.zeros(len(AA_LIST), dtype=np.float32)
        if resname in AA_TO_IDX:
            onehot[AA_TO_IDX[resname]] = 1.0
        node_feats.append(onehot)
        residue_keys.append((chain, resseq, resname))

    if not node_xyz:
        raise ValueError("No valid residues parsed from fpocket PDB.")

    X = np.vstack(node_xyz).astype(np.float32)
    F = np.vstack(node_feats).astype(np.float32)

    kdt = cKDTree(X)
    edges = set()

    if use_radius_edges:
        nbrs_list = kdt.query_ball_point(X, r=radius)
        for i, nbrs in enumerate(nbrs_list):
            for j in nbrs:
                if i != j:
                    edges.add((i, j))
    else:
        k = min(k + 1, len(X))
        _, idxs = kdt.query(X, k=k)
        for i, nbrs in enumerate(np.atleast_2d(idxs)):
            for j in nbrs:
                if i != j:
                    edges.add((i, j))

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    pos = torch.from_numpy(X)
    x = torch.from_numpy(F)

    if edge_index.numel() > 0:
        eu = X[edge_index[0].numpy()] - X[edge_index[1].numpy()]
        dist = np.linalg.norm(eu, axis=1, keepdims=True).astype(np.float32)
        edge_attr = torch.from_numpy(dist)
    else:
        edge_attr = torch.empty((0, 1), dtype=torch.float32)

    # Print proportion of nodes with degree >= 1 (has at least one incident edge)
    num_nodes = int(x.size(0))
    if edge_index.numel() > 0:
        nodes_with_edges = int(torch.unique(edge_index.view(-1)).numel())
    else:
        nodes_with_edges = 0
    prop = (nodes_with_edges / num_nodes) if num_nodes > 0 else 0.0
    print(f"[pocket] {Path(path).name}: nodes with deg>=1 = {nodes_with_edges}/{num_nodes} ({prop:.3f})")

    # Print normalized average degree (undirected) as a quick sparsity indicator
    # Build undirected neighbor sets to avoid double-counting directed edges
    if num_nodes > 0 and edge_index.numel() > 0:
        neigh = [set() for _ in range(num_nodes)]
        for u, v in edge_index.t().tolist():
            if u != v:
                neigh[u].add(v)
                neigh[v].add(u)
        avg_deg_undirected = sum(len(s) for s in neigh) / num_nodes
        norm_avg_deg = avg_deg_undirected / num_nodes
    else:
        avg_deg_undirected = 0.0
        norm_avg_deg = 0.0
    E = int(edge_index.size(1))
    print(f"[pocket] {Path(path).name}: avg_deg/N = {norm_avg_deg:.6f} (avg_deg={avg_deg_undirected:.3f}, N={num_nodes}, E={E})")

    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    data.residue_keys = residue_keys
    data.sequence_id = Path(path).stem
    return data
# pip install biopython numpy scipy torch torch_geometric
from Bio.PDB import PDBParser, MMCIFParser, is_aa
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch_geometric.data import Data

AA = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
      'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
AA2IDX = {a:i for i,a in enumerate(AA)}
BACKBONE = {'N','CA','C','O'}

def _load_structure(path, sid="prot"):
    if path.endswith((".cif",".mmcif")):
        return MMCIFParser(QUIET=True).get_structure(sid, path)
    return PDBParser(QUIET=True).get_structure(sid, path)

def _pseudo_cb_gly(res):
    try:
        N = res['N'].coord; CA = res['CA'].coord; C = res['C'].coord
        n = np.cross(N-CA, C-CA); n /= (np.linalg.norm(n) + 1e-8)
        return (CA + 1.5*n).astype(np.float32)
    except KeyError:
        return None

def _residue_coord(res, prefer_sidechain=True):
    atoms = [a for a in res if a.element and a.element.upper() != 'H']
    if not atoms: return None
    if res.get_resname().upper() == 'GLY':
        cb = _pseudo_cb_gly(res)
        if cb is not None: return cb
    if prefer_sidechain:
        sc = [a for a in atoms if a.get_name() not in BACKBONE]
        if sc:
            return np.vstack([a.coord for a in sc]).mean(0).astype(np.float32)
    # fallback: heavy-atom centroid, else CA
    try:
        return np.vstack([a.coord for a in atoms]).mean(0).astype(np.float32)
    except Exception:
        try: return res['CA'].coord.astype(np.float32)
        except KeyError: return None
        

def build_protein_graph_from_pdb(path: str,
                                 radius: float = 8.0,
                                 add_sequence_edges: bool = True) -> Data:
    """Residue graph with node=one residue; edges within `radius` Å + optional sequence edges."""
    s = _load_structure(path)
    node_xyz, node_feat, residue_keys, chain_seq = [], [], [], []

    for model in s:
        for chain in model:
            for res in chain:
                if not is_aa(res, standard=True):
                    continue
                xyz = _residue_coord(res, prefer_sidechain=True)
                if xyz is None:
                    continue
                node_xyz.append(xyz)
                onehot = np.zeros(len(AA), dtype=np.float32)
                rn = res.get_resname().upper()
                if rn in AA2IDX: onehot[AA2IDX[rn]] = 1.0
                # simple scalar: mean B-factor (optional; drop if you want exactly 20-D)
                bvals = [a.get_bfactor() for a in res if a.element and a.element.upper() != 'H']
                bmean = np.float32(np.mean(bvals) if bvals else 0.0)
                node_feat.append(np.concatenate([onehot, [bmean]], axis=0))  # 21 dims
                # include insertion code to uniquely identify residues
                icode = res.id[2] if len(res.id) > 2 else ' '
                residue_keys.append((chain.id, res.id[1], icode, rn))
                chain_seq.append((chain.id, res.id[1], len(node_xyz)-1))

    if not node_xyz:
        raise ValueError(f"No residues parsed from: {path}")

    X = np.vstack(node_xyz).astype(np.float32)       # [N,3]
    F = np.vstack(node_feat).astype(np.float32)      # [N,21]

    # spatial edges (undirected)
    kdt = cKDTree(X)
    nbrs = kdt.query_ball_point(X, r=radius)
    edges = set()
    for i, js in enumerate(nbrs):
        for j in js:
            if i != j:
                edges.add((i, j)); edges.add((j, i))

    # optional sequence edges (backbone continuity per chain)
    if add_sequence_edges:
        from collections import defaultdict
        per_chain = defaultdict(list)
        for cid, ridx, nid in ((c, r, n) for (c, r, _icode, _rn), n in zip(residue_keys, range(len(residue_keys)))):
            per_chain[cid].append((ridx, nid))
        for cid, lst in per_chain.items():
            lst.sort()
            for (_, i), (_, j) in zip(lst[:-1], lst[1:]):
                edges.add((i, j)); edges.add((j, i))

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()

    if edge_index.numel() > 0:
        diff = X[edge_index[0].numpy()] - X[edge_index[1].numpy()]
        dist = np.linalg.norm(diff, axis=1, keepdims=True).astype(np.float32)
    else:
        dist = np.empty((0,1), dtype=np.float32)

    # Print normalized average degree (undirected) for protein graph
    N = int(F.shape[0])
    E = int(edge_index.size(1))
    if N > 0 and E > 0:
        neigh = [set() for _ in range(N)]
        for u, v in edge_index.t().tolist():
            if u != v:
                neigh[u].add(v)
                neigh[v].add(u)
        avg_deg_undirected = sum(len(s) for s in neigh) / N
        norm_avg_deg = avg_deg_undirected / N
    else:
        avg_deg_undirected = 0.0
        norm_avg_deg = 0.0
    print(f"[protein] {Path(path).name}: avg_deg/N = {norm_avg_deg:.6f} (avg_deg={avg_deg_undirected:.3f}, N={N}, E={E})")

    data = Data(
        x=torch.from_numpy(F),           # [N, 21] (20 AA + mean B)
        pos=torch.from_numpy(X),         # [N, 3]
        edge_index=edge_index,           # [2, E]
        edge_attr=torch.from_numpy(dist) # [E, 1]  (RBF in the model)
    )
    data.residue_keys = residue_keys
    # Keep original file stem as protein_uid (e.g., may include 'id_')
    data.sequence_id = Path(path).stem.lower()
    return data

def featurize_proteins(
    protein_paths: List[str],
    radius: float = 8.0,
    num_workers: Optional[int] = None,
    show_progress: bool = True,
) -> List[Data]:
    """Parallel featurize proteins from PDB files into residue-distance graphs.

    - radius: distance threshold in Å for edges
    - num_workers: processes to use (default: cpu_count-1). Use 0/1 for serial.
    """
    total = len(protein_paths)
    if total == 0:
        return []

    # filter missing paths first
    paths = [str(p) for p in protein_paths if Path(p).exists()]
    missing = total - len(paths)
    if missing:
        print(f"[featurize_proteins] Skipped {missing} missing PDB paths out of {total}")

    out: List[Data] = []
    errors = 0
    if num_workers is None:
        cpu = os.cpu_count() or 2
        num_workers = max(1, cpu - 1)

    if num_workers <= 1:
        iterator = tqdm(paths, desc="proteins", total=len(paths)) if show_progress else paths
        for p in iterator:
            try:
                out.append(build_protein_graph_from_pdb(p, radius=radius))
            except Exception:
                errors += 1
        if errors:
            print(f"[featurize_proteins] Skipped {errors} malformed files")
        return out

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(build_protein_graph_from_pdb, p, radius) for p in paths]
        pbar = tqdm(total=len(futures), desc="proteins", disable=not show_progress)
        for fut in as_completed(futures):
            try:
                d = fut.result()
                out.append(d)
            except Exception:
                errors += 1
            finally:
                pbar.update(1)
        pbar.close()
    if errors:
        print(f"[featurize_proteins] Skipped {errors} malformed files")
    return out


def featurize_pockets(
    pocket_paths: List[str],
    k: int = 16,
    use_radius_edges: bool = False,
    radius: float = 6.0,
    num_workers: Optional[int] = None,
    show_progress: bool = True,
) -> List[Data]:
    """Parallel featurize pocket PDBs (fpocket outputs) into residue centroid graphs."""
    total = len(pocket_paths)
    if total == 0:
        return []

    paths = [str(p) for p in pocket_paths if Path(p).exists()]
    missing = total - len(paths)
    if missing:
        print(f"[featurize_pockets] Skipped {missing} missing PDB paths out of {total}")

    out: List[Data] = []
    errors = 0
    if num_workers is None:
        cpu = os.cpu_count() or 2
        num_workers = max(1, cpu - 1)

    if num_workers <= 1:
        iterator = tqdm(paths, desc="pockets", total=len(paths)) if show_progress else paths
        for p in iterator:
            try:
                out.append(build_pocket_graph_from_fpocket(p, k=k, use_radius_edges=use_radius_edges, radius=radius))
            except Exception:
                errors += 1
        if errors:
            print(f"[featurize_pockets] Skipped {errors} malformed files")
        return out

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(build_pocket_graph_from_fpocket, p, k, use_radius_edges, radius) for p in paths]
        pbar = tqdm(total=len(futures), desc="pockets", disable=not show_progress)
        for fut in as_completed(futures):
            try:
                d = fut.result()
                out.append(d)
            except Exception:
                errors += 1
            finally:
                pbar.update(1)
        pbar.close()
    if errors:
        print(f"[featurize_pockets] Skipped {errors} malformed files")
    return out

def one_hot_with_unknown(value, choices):
    idx = choices.index(value) if value in choices else len(choices)
    vec = torch.zeros(len(choices) + 1, dtype=torch.float32)
    vec[idx] = 1.0
    return vec

def one_hot_bool(flag: bool):
    """One-hot for boolean flags (no unknown bucket)."""
    return torch.tensor([1.0, 0.0], dtype=torch.float32) if not flag else torch.tensor([0.0, 1.0], dtype=torch.float32)


ATOM_LIST = [
"H","B","C","N","O","F","Si","P","S","Cl","Se","Br","I",
"Na","K","Mg","Ca","Al","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
"As","Sr","Zr","Mo","Ag","Cd","Sn","Sb","Te","Ba","W","Pt","Au","Hg","Pb","Bi"
]  # len=44

IMPL_VALENCE_LIST = [0, 1, 2, 3, 4, 5, 6]                 # + unknown
CHIRAL_LIST       = [CT.CHI_UNSPECIFIED, CT.CHI_TETRAHEDRAL_CW, CT.CHI_TETRAHEDRAL_CCW, CT.CHI_OTHER]  # + unknown
DEGREE_LIST       = [0, 1, 2, 3, 4, 5, 6]                 # + unknown
FORMAL_CHARGE_LIST= [-3, -2, -1, 0, 1, 2, 3]              # + unknown
NUM_HS_LIST       = [0, 1, 2, 3, 4]                       # + unknown
RADICAL_E_LIST    = [0, 1, 2]                              # + unknown
HYBRID_LIST       = [HT.S, HT.SP, HT.SP2, HT.SP3, HT.SP3D, HT.SP3D2, HT.UNSPECIFIED, HT.OTHER]


def atom_to_feature_vector(atom: Chem.rdchem.Atom) -> torch.Tensor:
    """ Concatenate one-hot vectors for the 10 attributes"""
    feats = [
    one_hot_with_unknown(atom.GetSymbol(), ATOM_LIST),
    one_hot_with_unknown(atom.GetImplicitValence(), IMPL_VALENCE_LIST),
    one_hot_with_unknown(atom.GetChiralTag(), CHIRAL_LIST),
    one_hot_with_unknown(atom.GetDegree(), DEGREE_LIST),
    one_hot_with_unknown(atom.GetFormalCharge(), FORMAL_CHARGE_LIST),
    one_hot_with_unknown(atom.GetTotalNumHs(), NUM_HS_LIST),
    one_hot_with_unknown(atom.GetNumRadicalElectrons(), RADICAL_E_LIST),
    one_hot_with_unknown(atom.GetHybridization(), HYBRID_LIST),
    one_hot_bool(atom.GetIsAromatic()),
    one_hot_bool(atom.IsInRing()),
    ]
    return torch.cat(feats, dim=0)


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """Convert a SMILES string into a torch_geometric Data graph with atom features.

    Returns None if the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n = mol.GetNumAtoms()
    x = torch.stack([atom_to_feature_vector(a) for a in mol.GetAtoms()], dim=0)
    rows: List[int] = []
    cols: List[int] = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        rows += [i, j]
        cols += [j, i]
    edge_index = (
        torch.tensor([rows, cols], dtype=torch.long)
        if rows
        else torch.empty((2, 0), dtype=torch.long)
    )
    d = Data(x=x, edge_index=edge_index, num_nodes=n)
    d.ligand_id = smiles  # keep a stable id matching CSV 'Ligand'
    return d


def featurize_ligands(smiles_list: List[str]) -> List[Data]:
    """Vectorize a list of SMILES into torch_geometric graphs."""
    data_list: List[Data] = []
    for smi in smiles_list:
        d = smiles_to_graph(smi)
        if d is not None:
            data_list.append(d)
    return data_list

def graph_embedder(
    smiles_list: List[str],
    out_path: Optional[str] = None,
    save_as_collated: bool = False,
) -> Union[List[Data], Data]:
    """Featurize SMILES and optionally save them.

    - If save_as_collated=False: returns List[Data]; optionally saves list to out_path.
    - If save_as_collated=True: returns a Batch; saves a single packed Data object.
    """
    data_list = featurize_ligands(smiles_list)
    if save_as_collated:
        packed = Batch.from_data_list(data_list)
        if out_path:
            torch.save(packed, out_path)
        return packed
    else:
        if out_path:
            torch.save(data_list, out_path)
        return data_list


    # =============== DGraphDTA-style protein featurization (sequence + PSSM + contact map) ===============
    # This replicates the core idea from DGraphDTA: per-residue features (one-hot + physicochemical props)
    # concatenated with PSSM (from an alignment file), and edges from a predicted contact map (pconsc4).

    # Amino-acid table including 'X' for unknowns
    DGRAPH_RES_TABLE = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','X']

    _ALIPHATIC = {'A','I','L','M','V'}
    _AROMATIC = {'F','W','Y'}
    _POLAR_NEUTRAL = {'C','N','Q','S','T'}
    _ACIDIC = {'D','E'}
    _BASIC = {'H','K','R'}

    _RES_WEIGHT = {'A':71.08,'C':103.15,'D':115.09,'E':129.12,'F':147.18,'G':57.05,'H':137.14,'I':113.16,'K':128.18,'L':113.16,'M':131.20,'N':114.11,'P':97.12,'Q':128.13,'R':156.19,'S':87.08,'T':101.11,'V':99.13,'W':186.22,'Y':163.18}
    _PKA = {'A':2.34,'C':1.96,'D':1.88,'E':2.19,'F':1.83,'G':2.34,'H':1.82,'I':2.36,'K':2.18,'L':2.36,'M':2.28,'N':2.02,'P':1.99,'Q':2.17,'R':2.17,'S':2.21,'T':2.09,'V':2.32,'W':2.83,'Y':2.32}
    _PKB = {'A':9.69,'C':10.28,'D':9.60,'E':9.67,'F':9.13,'G':9.60,'H':9.17,'I':9.60,'K':8.95,'L':9.60,'M':9.21,'N':8.80,'P':10.60,'Q':9.13,'R':9.04,'S':9.15,'T':9.10,'V':9.62,'W':9.39,'Y':9.62}
    _PKX = {'A':0.00,'C':8.18,'D':3.65,'E':4.25,'F':0.00,'G':0.00,'H':6.00,'I':0.00,'K':10.53,'L':0.00,'M':0.00,'N':0.00,'P':0.00,'Q':0.00,'R':12.48,'S':0.00,'T':0.00,'V':0.00,'W':0.00,'Y':0.00}
    _PL = {'A':6.00,'C':5.07,'D':2.77,'E':3.22,'F':5.48,'G':5.97,'H':7.59,'I':6.02,'K':9.74,'L':5.98,'M':5.74,'N':5.41,'P':6.30,'Q':5.65,'R':10.76,'S':5.68,'T':5.60,'V':5.96,'W':5.89,'Y':5.96}
    _HYDRO_PH2 = {'A':47,'C':52,'D':-18,'E':8,'F':92,'G':0,'H':-42,'I':100,'K':-37,'L':100,'M':74,'N':-41,'P':-46,'Q':-18,'R':-26,'S':-7,'T':13,'V':79,'W':84,'Y':49}
    _HYDRO_PH7 = {'A':41,'C':49,'D':-55,'E':-31,'F':100,'G':0,'H':8,'I':99,'K':-23,'L':97,'M':74,'N':-28,'P':-46,'Q':-10,'R':-14,'S':-5,'T':13,'V':76,'W':97,'Y':63}

    def _normalize_prop_table(d: dict) -> dict:
        vals = list(d.values())
        mx, mn = float(max(vals)), float(min(vals))
        interval = (mx - mn) if mx > mn else 1.0
        out = {k: (float(v) - mn) / interval for k, v in d.items()}
        out['X'] = (mx + mn) / 2.0
        return out

    # Pre-normalize continuous property tables
    _RES_WEIGHT = _normalize_prop_table(_RES_WEIGHT)
    _PKA = _normalize_prop_table(_PKA)
    _PKB = _normalize_prop_table(_PKB)
    _PKX = _normalize_prop_table(_PKX)
    _PL = _normalize_prop_table(_PL)
    _HYDRO_PH2 = _normalize_prop_table(_HYDRO_PH2)
    _HYDRO_PH7 = _normalize_prop_table(_HYDRO_PH7)

    def _residue_property_vector(aa: str) -> np.ndarray:
        aa = aa if aa in DGRAPH_RES_TABLE else 'X'
        props_cat = [
            1 if aa in _ALIPHATIC else 0,
            1 if aa in _AROMATIC else 0,
            1 if aa in _POLAR_NEUTRAL else 0,
            1 if aa in _ACIDIC else 0,
            1 if aa in _BASIC else 0,
        ]
        props_cont = [
            _RES_WEIGHT.get(aa, _RES_WEIGHT['X']),
            _PKA.get(aa, _PKA['X']),
            _PKB.get(aa, _PKB['X']),
            _PKX.get(aa, _PKX['X']),
            _PL.get(aa, _PL['X']),
            _HYDRO_PH2.get(aa, _HYDRO_PH2['X']),
            _HYDRO_PH7.get(aa, _HYDRO_PH7['X']),
        ]
        return np.array(props_cat + props_cont, dtype=np.float32)  # 12 dims

    def _one_hot_aa_with_X(aa: str) -> np.ndarray:
        aa = aa if aa in DGRAPH_RES_TABLE else 'X'
        vec = np.zeros(len(DGRAPH_RES_TABLE), dtype=np.float32)
        vec[DGRAPH_RES_TABLE.index(aa)] = 1.0
        return vec

    def _seq_feature(seq: str) -> np.ndarray:
        L = len(seq)
        hot = np.zeros((L, len(DGRAPH_RES_TABLE)), dtype=np.float32)
        prop = np.zeros((L, 12), dtype=np.float32)
        for i, ch in enumerate(seq):
            hot[i] = _one_hot_aa_with_X(ch)
            prop[i] = _residue_property_vector(ch)
        return np.concatenate([hot, prop], axis=1)  # [L, 33]

    def _pssm_from_aln(aln_path: str, seq: str) -> np.ndarray:
        # Build a position-frequency matrix from alignment characters (using the same 21 AA table + X)
        if not os.path.exists(aln_path):
            # fallback: zeros
            return np.zeros((len(seq), len(DGRAPH_RES_TABLE)), dtype=np.float32)
        with open(aln_path, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if len(lines) == 0:
            return np.zeros((len(seq), len(DGRAPH_RES_TABLE)), dtype=np.float32)
        pfm = np.zeros((len(DGRAPH_RES_TABLE), len(seq)), dtype=np.float32)
        line_count = 0
        for line in lines:
            if len(line) != len(seq):
                # skip malformed lines
                continue
            line_count += 1
            for i, ch in enumerate(line):
                aa = ch if ch in DGRAPH_RES_TABLE else 'X'
                pfm[DGRAPH_RES_TABLE.index(aa), i] += 1.0
        pseudocount = 0.8
        denom = (float(line_count) + pseudocount)
        if denom <= 0:
            denom = 1.0
        ppm = (pfm + pseudocount / 4.0) / denom
        # Return [L, 21] to match concatenation with other features
        return np.transpose(ppm, (1, 0)).astype(np.float32)

    def build_protein_graph_from_dgraph(
        target_key: str,
        target_sequence: str,
        contact_dir: str,
        aln_dir: str,
        contact_threshold: float = 0.5,
    ) -> Data:
        """Create a residue graph using pconsc4 contacts and DGraphDTA features.

        Node features: concat([PSSM(Lx21), one-hot+physicochemical(Lx33)]) -> [L, 54]
        Edges: indices where contact_map >= threshold (self-loops included if present)
        """
        L = len(target_sequence)
        # features
        other = _seq_feature(target_sequence)              # [L, 33]
        aln_path = os.path.join(aln_dir, f"{target_key}.aln")
        pssm = _pssm_from_aln(aln_path, target_sequence)  # [L, 21]
        feat = np.concatenate([pssm, other], axis=1).astype(np.float32)  # [L, 54]

        # edges from contact map
        cpath = os.path.join(contact_dir, f"{target_key}.npy")
        if not os.path.exists(cpath):
            # empty graph with no edges
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            contact = np.load(cpath)
            # add identity to keep at least self indices consistent (as in DGraphDTA)
            contact = contact + np.eye(contact.shape[0], dtype=contact.dtype)
            idx_r, idx_c = np.where(contact >= contact_threshold)
            # Use directed edges (PyG style); filter indices within [0, L)
            mask = (idx_r < L) & (idx_c < L)
            rows = idx_r[mask].astype(np.int64)
            cols = idx_c[mask].astype(np.int64)
            if rows.size == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor([rows, cols], dtype=torch.long)

        d = Data(
            x=torch.from_numpy(feat),           # [L, 54]
            edge_index=edge_index,              # [2, E]
        )
        d.protein_uid = str(target_key)
        d.sequence_id = str(target_key)
        return d

    def featurize_proteins_dgraph(
        keys: List[str],
        sequences: List[str],
        contact_dir: str,
        aln_dir: str,
        show_progress: bool = True,
    ) -> List[Data]:
        assert len(keys) == len(sequences), "keys and sequences must be same length"
        out: List[Data] = []
        iterator = zip(keys, sequences)
        if show_progress:
            iterator = tqdm(list(iterator), total=len(keys), desc="proteins(dgraph)")
        for key, seq in iterator:
            try:
                out.append(build_protein_graph_from_dgraph(key, seq, contact_dir, aln_dir))
            except Exception:
                pass
        return out