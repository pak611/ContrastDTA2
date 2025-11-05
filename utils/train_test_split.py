"""
Simple utilities to add a 'split' column (train/val/test) to a pandas DataFrame.

Usage:
    from utils.train_test_split import add_split_column
    df = add_split_column(df, train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=42)

Notes:
    - This is a minimal, no-frills splitter: it shuffles rows and assigns
      fractions by order. No stratification or grouping.
    - Fractions should sum to 1.0. The last split receives the remainder
      to ensure all rows are assigned.
"""

from __future__ import annotations

import pandas as pd


def add_split_column(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    inplace: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame with a new 'split' column of values in {train,val,test}.

    This function shuffles the rows deterministically and assigns the first
    train_frac rows to 'train', the next val_frac rows to 'val', and the
    remainder to 'test'.
    """
    n = len(df)
    if not inplace:
        df = df.copy()

    # Deterministic shuffle
    shuffled = df.sample(frac=1.0, random_state=seed).index

    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    # Assign the remainder to test to cover rounding
    n_test = n - n_train - n_val

    train_idx = shuffled[:n_train]
    val_idx = shuffled[n_train:n_train + n_val]
    test_idx = shuffled[n_train + n_val:]

    df.loc[train_idx, 'split'] = 'train'
    df.loc[val_idx, 'split'] = 'val'
    df.loc[test_idx, 'split'] = 'test'

    return df


__all__ = ["add_split_column"]
