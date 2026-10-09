"""Safety projection onto {0 <= p <= h, sum p <= P_grid}.

uniform: p = clip(p~ - mu, 0, h), mu >= 0 smallest value with sum p <= P_grid (exact Euclidean projection).
priority: same bisection, but the cut is mu * w_i with w_i >= 0 (low-laxity vehicles get small w_i, so they are
cut less). Same interface, selected by `scheme` (E5 ablation). Not a Euclidean projection but always feasible.
"""
import numpy as np


def project(p_tilde, h, P_grid, scheme="uniform", weights=None, iters=60):
    p_tilde = np.asarray(p_tilde, dtype=float)
    h = np.asarray(h, dtype=float)
    p0 = np.clip(p_tilde, 0.0, h)
    if p0.sum() <= P_grid:
        return p0
    if scheme == "uniform":
        w = np.ones_like(p0)
    elif scheme == "priority":
        w = np.asarray(weights, dtype=float)
    else:
        raise ValueError(f"unknown projection scheme {scheme!r}")
    # sum(clip(p~ - mu*w, 0, h)) is continuous and non-increasing in mu; it is 0 once mu*w >= p~ for all w>0
    lo, hi = 0.0, float(np.max(p0 / np.maximum(w, 1e-12)))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if np.clip(p_tilde - mid * w, 0.0, h).sum() > P_grid:
            lo = mid
        else:
            hi = mid
    return np.clip(p_tilde - hi * w, 0.0, h)  # hi side is always feasible
