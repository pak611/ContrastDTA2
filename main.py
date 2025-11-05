# main.py (fixed essentials)

from __future__ import annotations
import os, sys
from pathlib import Path
import argparse
import pandas as pd
import torch
import yaml
import json

from torch_geometric.loader import DataLoader  # <- PyG loader
from utils.featurizer import (
    featurize_proteins, featurize_ligands, build_pocket_graph_from_fpocket, featurize_pockets,
    featurize_proteins_dgraph
)
from utils.dataset import PLDataset, PLDataset2  # pocket or protein variants
from utils.train_test_split import add_split_column
from models.base_model import BaseDTA, BaseDTA2
from models.pocket_net import PocketEncoder
from models.protein_net import ProteinNet
from models.ligand_net import LigandNet
from models.trainer import Trainer
from utils.metrics import compute_regression_metrics

parser = argparse.ArgumentParser(description="DTI Contrastive Model Demo")
parser.add_argument('--config', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--processed_path', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--input', type=str, required=False)
parser.add_argument('--pocket_path', type=str, required=False)
parser.add_argument('--protein_path', type=str, required=False)
parser.add_argument('--protein_feature_mode', type=str, choices=['pdb','dgraph'], default='pdb', help='protein featurization mode')
parser.add_argument('--contact_dir', type=str, required=False, help='dir containing pconsc4 .npy files (for dgraph mode)')
parser.add_argument('--aln_dir', type=str, required=False, help='dir containing alignment .aln files (for dgraph mode)')
parser.add_argument('--protein_key_col', type=str, default='ID', help='CSV column for protein key (for dgraph mode)')
parser.add_argument('--protein_seq_col', type=str, default='ProteinSequence', help='CSV column for protein sequence (for dgraph mode)')
# paths for cached features
parser.add_argument('--processed_protein_path', type=str, default='processed_proteins.pt')
parser.add_argument('--processed_ligand_path',  type=str, default='processed_ligands.pt')
parser.add_argument('--processed_pocket_path',  type=str, default='processed_pockets.pt')
# training
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--model_type', type=str, choices=['pocket','protein'], default=None, help='pocket or protein')
parser.add_argument('--workers', type=int, default=None, help='parallel workers for featurization (default: cpu_count-1)')
## no contrast / grouped sampling CLI in this revert
args = parser.parse_args()

# Load config overrides
if args.config:
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    for section in cfg:
        for k, v in cfg[section].items():
            setattr(args, k, v)

def main() -> None:
    # --- 1) Read CSV + split
    df = pd.read_csv(args.input)
    df = add_split_column(df, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42)

    ligand_smiles = df['Ligand'].astype(str).tolist()
    # Determine model type from args or config
    model_type = getattr(args, 'model_type', None) or getattr(args, 'type', None) or 'pocket'
    if model_type not in ('pocket', 'protein'):
        model_type = 'pocket'

    # --- 2) Load / build features
    # Load/build proteins only if protein model selected
    if model_type == 'protein':
        if os.path.exists(args.processed_protein_path):
            print("Loading cached protein graphs…")
            protein_data = torch.load(args.processed_protein_path, weights_only=False)
        else:
            if args.protein_feature_mode == 'pdb':
                if not args.protein_path:
                    raise ValueError("protein_path is required when model_type=protein (pdb) and no cached proteins found")
                protein_dir = Path(args.protein_path)
                pdb_files = sorted(str(p) for p in protein_dir.rglob("*.pdb"))
                print(f"Featurizing {len(pdb_files)} proteins from: {protein_dir}")
                protein_data = featurize_proteins(pdb_files, radius=8.0, num_workers=args.workers)
                torch.save(protein_data, args.processed_protein_path)
            else:  # dgraph mode
                key_col = args.protein_key_col
                seq_col = args.protein_seq_col
                if key_col not in df.columns or seq_col not in df.columns:
                    raise ValueError(f"CSV must contain columns '{key_col}' and '{seq_col}' for dgraph featurization")
                if not args.contact_dir or not args.aln_dir:
                    raise ValueError("contact_dir and aln_dir are required for dgraph featurization")
                # use unique keys -> one sequence per key (take first occurrence)
                key_to_seq = {}
                for key, seq in zip(df[key_col].astype(str).tolist(), df[seq_col].astype(str).tolist()):
                    if key not in key_to_seq:
                        key_to_seq[key] = seq
                keys = list(key_to_seq.keys())
                seqs = [key_to_seq[k] for k in keys]
                print(f"Featurizing {len(keys)} proteins (dgraph) from contact_dir={args.contact_dir}")
                protein_data = featurize_proteins_dgraph(keys, seqs, args.contact_dir, args.aln_dir)
                torch.save(protein_data, args.processed_protein_path)
    else:
        protein_data = []

    if os.path.exists(args.processed_ligand_path):
        print("Loading cached ligands…")
        ligand_data = torch.load(args.processed_ligand_path, weights_only=False)
    else:
        print("Featurizing ligands…")
        ligand_data = featurize_ligands(ligand_smiles)
        torch.save(ligand_data, args.processed_ligand_path)

    if os.path.exists(args.processed_pocket_path):
        print("Loading cached pockets…")
        pocket_data = torch.load(args.processed_pocket_path, weights_only=False)
    else:
        pocket_data = []
        if args.pocket_path:
            pocket_dir = Path(args.pocket_path)
            if pocket_dir.exists():
                pdb_files = sorted(str(p) for p in pocket_dir.rglob("*.pdb"))
                print(f"Featurizing {len(pdb_files)} pockets from: {pocket_dir}")
                pocket_data = featurize_pockets(pdb_files, num_workers=args.workers)
                torch.save(pocket_data, args.processed_pocket_path)

    # --- 3) Build ID maps
    ligand_map  = {g.ligand_id: g for g in ligand_data}
    # Map by both the original stem (protein_uid) and the normalized sequence_id (without 'id_')
    protein_map = {}
    for p in protein_data:
        sid = getattr(p, 'sequence_id', None)
        protein_map[sid] = p

        
    pocket_map  = {g.sequence_id: g for g in pocket_data}

    # --- 4) Build all rows, keep meta out of HeteroData
    rows_all = []
    skipped = 0
    for row in df.itertuples(index=False):
        try:
            L = ligand_map[row.Ligand]
            y = float(row.regression_label)
            meta = {"split": row.split, "id": row.ID}
            if model_type == 'pocket':
                P = pocket_map[row.ID]
            else:
                # In dgraph mode, the protein key column might differ; map with the configured key
                if args.protein_feature_mode == 'dgraph':
                    key_val = getattr(row, args.protein_key_col)
                    P = protein_map[str(key_val)]
                else:
                    P = protein_map[row.ID]
            rows_all.append((P, L, y, meta))
        except KeyError:
            skipped += 1
    if skipped:
        print(f"Skipped {skipped} rows due to missing keys")

    rows_train = [r for r in rows_all if r[3]['split'] == 'train']
    rows_val   = [r for r in rows_all if r[3]['split'] == 'val']
    rows_test  = [r for r in rows_all if r[3]['split'] == 'test']

    # --- 5) Datasets & Loaders  (PLDataset must put edges into EDGE STORES)
    # Ensure ligand edge_attr exists (LigandNet expects edge features)
    # Infer ligand node/edge dims from cached graphs
    sample_lig = next(iter(ligand_map.values()))
    ligand_node_in = int(sample_lig.x.size(1))
    ligand_edge_in = int(getattr(sample_lig, 'edge_attr', torch.ones((1,1))).size(1) if hasattr(sample_lig, 'edge_attr') and sample_lig.edge_attr is not None else 1)

    def ensure_ligand_edge_attr(d):
        est = d[("ligand", "ligand")]
        # ensure edge_attr exists and is a tensor; handle attribute missing or None
        if (not hasattr(est, 'edge_attr')) or (getattr(est, 'edge_attr', None) is None):
            E = est.edge_index.size(1)
            est.edge_attr = torch.ones((E, ligand_edge_in), dtype=torch.float32)
        return d

    if model_type == 'pocket':
        train_ds = PLDataset(rows_train, transform=ensure_ligand_edge_attr)
        val_ds   = PLDataset(rows_val,   transform=ensure_ligand_edge_attr)
        test_ds  = PLDataset(rows_test,  transform=ensure_ligand_edge_attr)
    else:
        train_ds = PLDataset2(rows_train, transform=ensure_ligand_edge_attr)
        val_ds   = PLDataset2(rows_val,   transform=ensure_ligand_edge_attr)
        test_ds  = PLDataset2(rows_test,  transform=ensure_ligand_edge_attr)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # --- 6) Model & Trainer
    device = torch.device(args.device)
    # Determine latent dimension from config (fallback 128)
    d_latent = int(getattr(args, 'd_model', 128))
    print("Using d_latent =", d_latent)
    gnn_layers = int(getattr(args, 'gnn_layers', 3))
    n_heads = int(getattr(args, 'n_heads', 4))
    dropout = float(getattr(args, 'dropout', 0.1))
    # Infer protein node feature dim if needed (protein graphs are typically 21: 20 AA + B-factor)
    protein_node_in = 21
    if model_type == 'protein' and len(protein_data) > 0:
        try:
            protein_node_in = int(getattr(protein_data[0], 'x').size(1))
        except Exception:
            protein_node_in = 21

    ligand_encoder = LigandNet(
        node_in=ligand_node_in,
        edge_in=ligand_edge_in,
        hidden=d_latent,
        layers=gnn_layers,
        out_dim=d_latent,
        dropout=dropout,
    )
    if model_type == 'pocket':
        pocket_encoder = PocketEncoder(node_in=20, hidden=d_latent, layers=gnn_layers)
        model = BaseDTA(
            pocket_encoder=pocket_encoder,
            ligand_encoder=ligand_encoder,
            d_latent=d_latent,
            n_heads=n_heads,
            dropout=dropout,
        ).to(device)
    else:
        protein_encoder = ProteinNet(node_in=protein_node_in, hidden=d_latent, layers=gnn_layers, out_dim=d_latent, dropout=dropout)
        model = BaseDTA2(
            protein_encoder=protein_encoder,
            ligand_encoder=ligand_encoder,
            d_latent=d_latent,
            dropout=dropout,
        ).to(device)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
        device=device,
        log_interval=getattr(args, "log_interval", 50),
    )

    # --- Prepare results dir
    results_dir = Path(__file__).resolve().parent / "results"
    os.makedirs(results_dir, exist_ok=True)

    # --- 7) Training loop (no re-creating loaders inside the loop)
    epoch_metrics = []
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        trainer.train_epoch()

        # quick val pass (if Trainer doesn't already do it)
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)        # <-- move to device
                out = model(batch)              # <-- pass the HeteroData batch
                preds.append(out.detach().cpu())
                targets.append(batch.y.detach().cpu())
        if preds:
            yp = torch.cat(preds).view(-1).numpy()
            yt = torch.cat(targets).view(-1).numpy()
            metrics = compute_regression_metrics(yt, yp)
            metrics_row = {"epoch": epoch + 1, **metrics}
            epoch_metrics.append(metrics_row)
            # print nicely to console
            print(
                "val epoch {ep}: "
                "rmse={rmse:.4f}, mse={mse:.4f}, ci={ci:.4f}, "
                "pearson={pearson:.4f}, spearman={spearman:.4f}, r2m={r2m:.4f}".format(
                    ep=epoch + 1,
                    rmse=metrics.get("rmse", float("nan")),
                    mse=metrics.get("mse", float("nan")),
                    ci=metrics.get("ci", float("nan")),
                    pearson=metrics.get("pearson", float("nan")),
                    spearman=metrics.get("spearman", float("nan")),
                    r2m=metrics.get("r2m", float("nan")),
                )
            )
            # write/refresh epoch metrics file each epoch
            with open(results_dir / "epoch_metrics.json", "w") as f:
                json.dump(epoch_metrics, f, indent=2)

        # Save a small collection of interaction maps from the val set each epoch (if supported)
        try:
            maps_dir = results_dir / "interaction_maps"
            os.makedirs(maps_dir, exist_ok=True)
            # save up to 8 maps from the first few val batches
            from models.trainer import Trainer as _T  # just to satisfy type hints if any
            trainer.save_interaction_maps(val_loader, str(maps_dir), epoch + 1, limit=8)
        except Exception:
            # Non-fatal: proceed even if map saving isn't supported
            pass

    # --- 8) Final test (optional)
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            test_preds.append(out.detach().cpu())
            test_targets.append(batch.y.detach().cpu())
    if test_preds:
        yp = torch.cat(test_preds).view(-1).numpy()
        yt = torch.cat(test_targets).view(-1).numpy()
        test_metrics = compute_regression_metrics(yt, yp)
        with open(results_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)
        # predictions.json as list of objects with regression_label and predicted_label
        pred_rows = [
            {"regression_label": float(t), "predicted_label": float(p)}
            for t, p in zip(yt.tolist(), yp.tolist())
        ]
        with open(results_dir / "predictions.json", "w") as f:
            json.dump(pred_rows, f, indent=2)

if __name__ == "__main__":
    main()
