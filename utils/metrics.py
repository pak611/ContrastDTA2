import math
from typing import Dict, Iterable
import numpy as np


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Tie-aware average ranks (1-based), similar to scipy.stats.rankdata(method='average')."""
    a = np.asarray(a)
    uniques, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    cumsum = np.cumsum(counts)
    starts = cumsum - counts
    # average rank for each unique value in 0-based indexing
    avg_ranks0 = (starts + cumsum - 1) / 2.0
    # convert to 1-based ranks
    return avg_ranks0[inverse] + 1.0


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float('nan')
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.linalg.norm(xm) * np.linalg.norm(ym)
    if denom == 0:
        return float('nan')
    return float(np.dot(xm, ym) / denom)


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _pearsonr(rx, ry)


def _concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Concordance index for regression rankings.

    Counts pairs (i<j) with y_i != y_j. Concordant if prediction order matches true order.
    Ties in prediction count as 0.5 if true order is unequal.
    """
    n = y_true.shape[0]
    conc = 0.0
    ties = 0.0
    total = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            total += 1.0
            dp = y_pred[i] - y_pred[j]
            dt = y_true[i] - y_true[j]
            if dp == 0:
                ties += 1.0
            elif (dp > 0 and dt > 0) or (dp < 0 and dt < 0):
                conc += 1.0
    if total == 0:
        return float('nan')
    return float((conc + 0.5 * ties) / total)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = y_true
    yp = y_pred
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot == 0:
        return float('nan')
    return 1.0 - ss_res / ss_tot


def _r2_origin(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # fit y = kx through origin
    x = y_pred
    y = y_true
    denom = float(np.dot(x, x))
    k = float(np.dot(x, y) / denom) if denom != 0 else 0.0
    y0 = k * x
    ss_res = float(np.sum((y - y0) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return float('nan')
    return 1.0 - ss_res / ss_tot


def compute_regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_pred), dtype=float)
    mse = float(np.mean((yt - yp) ** 2))
    rmse = float(math.sqrt(mse))
    pearson = _pearsonr(yt, yp)
    spearman = _spearmanr(yt, yp)
    ci = _concordance_index(yt, yp)
    r2 = _r2_score(yt, yp)
    r0_2 = _r2_origin(yt, yp)
    # common r_m^2 definition
    try:
        r2m = float(r2 * (1.0 - math.sqrt(abs(r2 - r0_2))))
    except ValueError:
        r2m = float('nan')
    return {
        "mse": mse,
        "rmse": rmse,
        "pearson": float(pearson),
        "spearman": float(spearman),
        "ci": float(ci),
        "r2m": float(r2m),
    }
