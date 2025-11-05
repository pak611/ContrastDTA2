# ContrastDTA2: Contrastive DTI MVP

This folder contains a minimal, runnable scaffold for contrastive ligand–protein representation learning with optional cross-attention and a pocket encoder path.

## Components
- `models/dense_gnn.py` — Simple dense GNN (no torch-geometric) with mask-aware pooling for ligands.
- `models/transformer.py` — Minimal Transformer encoder and a cross-attention block for sequences.
- `models/projection.py` — Small 2-layer projection head with LayerNorm and L2-normalization.
- `models/contrastive.py` — Symmetric InfoNCE with learnable temperature.
- `models/dti_contrastive.py` — Wires encoders, optional cross-attention, and the contrastive head.
- `utils/pocket.py` — Biopython-based pocket PDB -> residue-graph constructor.
- `main.py` — Toy demo that builds random ligand graphs and protein token sequences and computes the contrastive loss.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Expected output (example):
```
z_ligand: (8, 128)
z_protein: (8, 128)
loss: 2.3
```

## Pocket encoder usage (optional)
If you have pocket PDBs extracted (e.g., via fpocket), you can turn them into residue graphs:

```python
from utils.pocket import load_pocket_residue_graph
X, C, A = load_pocket_residue_graph("path/to/pocket.pdb", r_edge=10.0)
# X: [N, 20], C: [N, 3], A: [N, N]
```

You can adapt `DTIContrastiveModel` to take pocket graphs instead of full-protein sequences by encoding X/A through the ligand GNN (or a separate residue-GNN) and using that pooled vector for contrastive training or cross-attention.

## Notes
- This MVP avoids heavy deps (torch-geometric, ESM). Swap in your preferred GNN/ESM later.
- All masking follows the convention: `True` means padding.
- Cross-attention uses `nn.MultiheadAttention` with batch_first=True.
# ContrastDTA2: Simple Transformer Demo

This folder contains a minimal, flexible Transformer encoder implemented in PyTorch and a small demo runner.

## Files
- models/simple_transformer.py — Compact Transformer encoder with optional token embedding or feature projection, sinusoidal positional encoding, pooling, and a small output head.
- main.py — Builds a toy batch of token sequences with padding, runs a forward pass, and prints output shapes.
- requirements.txt — Minimal dependency list (PyTorch).

## Quick start

```
# (optional) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# run the demo
python main.py
```

Expected output (shapes may vary):
```
Output y shape: (4, 1)
Pooled shape: (4, 64)
Hidden shape: (4, 21, 64)
```

Notes:
- If CUDA is available, the model will use it automatically; otherwise it runs on CPU.
- You can switch to continuous feature inputs by initializing SimpleTransformer with input_dim=... instead of vocab_size and passing tensors of shape [B, L, input_dim].