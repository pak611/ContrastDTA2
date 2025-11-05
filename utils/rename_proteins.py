#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path
import shutil


def load_mapping(csv_path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with csv_path.open(newline='') as f:
        reader = csv.DictReader(f)
        # Expect headers: sequence_id,pdb_id
        if 'sequence_id' not in reader.fieldnames or 'pdb_id' not in reader.fieldnames:
            raise ValueError(f"CSV {csv_path} must have columns: sequence_id,pdb_id; got {reader.fieldnames}")
        for r in reader:
            seq = r['sequence_id'].strip()
            pdb = r['pdb_id'].strip()
            if seq and pdb:
                rows.append((seq, pdb))
    return rows


def main(proteins_dir: str, mapping_csv: str, mode: str = 'copy') -> None:
    src_dir = Path(proteins_dir)
    csv_path = Path(mapping_csv)
    if not src_dir.is_dir():
        raise SystemExit(f"proteins_dir not found: {src_dir}")
    if not csv_path.is_file():
        raise SystemExit(f"mapping CSV not found: {csv_path}")

    rows = load_mapping(csv_path)
    created = 0
    skipped_exist = 0
    missing_src = 0
    missing_rows: list[tuple[str, str]] = []  # (sequence_id, pdb_id)

    for seq_id, pdb_id in rows:
        # Normalize case; proteins appear to be stored as UPPER with .pdb
        src_candidates = [
            src_dir / f"{pdb_id.upper()}.pdb",
            src_dir / f"{pdb_id.lower()}.pdb",
            src_dir / f"{pdb_id}.pdb",
        ]
        src = next((p for p in src_candidates if p.exists()), None)
        if src is None:
            missing_src += 1
            missing_rows.append((seq_id, pdb_id))
            continue

        dst = src_dir / f"{seq_id}.pdb"
        if dst.exists():
            skipped_exist += 1
            continue

        try:
            if mode == 'link':
                # Hardlink to save space when possible
                dst.hardlink_to(src)
            else:
                shutil.copy2(src, dst)
            created += 1
        except Exception:
            # Fallback to copy if hardlink fails
            try:
                shutil.copy2(src, dst)
                created += 1
            except Exception as e:
                print(f"Failed to create {dst.name} from {src.name}: {e}")

    total = len(rows)
    # Write a report of missing sources
    report_path = src_dir / "rename_report.txt"
    with report_path.open('w') as rf:
        rf.write(f"total_mappings: {total}\n")
        rf.write(f"created: {created}\n")
        rf.write(f"existing_skipped: {skipped_exist}\n")
        rf.write(f"missing_sources: {missing_src}\n")
        if missing_rows:
            rf.write("\n# Missing source PDBs (sequence_id,pdb_id)\n")
            for seq_id, pdb in missing_rows[:1000]:
                rf.write(f"{seq_id},{pdb}\n")

    print(f"Done: total mappings={total}, created={created}, existing_skipped={skipped_exist}, missing_sources={missing_src}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: rename_proteins.py <proteins_dir> <sequence_to_pdb.csv> [copy|link]")
        raise SystemExit(2)
    proteins_dir = sys.argv[1]
    mapping_csv = sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else 'copy'
    main(proteins_dir, mapping_csv, mode)
