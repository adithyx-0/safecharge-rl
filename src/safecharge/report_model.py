"""Report modes M0 (truthful), M2 (noisy), M3 (late), M4 (strategic); M1r = recorded driver reports (supplementary, see below).

Draws are made once per (seed, day) from a fixed-size base stream that does not depend on mode or parameters,
so every method sees the same draws, and a larger w / q / rho reuses the same underlying numbers (nested).
Reports are in steps (float) and clipped to stay at or after arrival. Times are in steps; hours / DELTA.
"""
import numpy as np

DELTA = 0.25


def base_draws(seed, day_index, n):
    rng = np.random.default_rng([seed, day_index])
    return dict(eps=rng.uniform(-1, 1, n), late=rng.uniform(0, 1, n), sel=rng.uniform(0, 1, n), imp=rng.uniform(0, 1, n))


def empirical_errors(days):
    """Pool of real recorded errors (recorded departure - true departure, in steps) from `days`, for M1r imputation."""
    e = np.concatenate([d.rep_rec - d.dep for d in days])
    return e[~np.isnan(e)]


def reports(day, mode, seed, day_index, w=0.0, q=0.0, rho=0.0, s=0.0, focal=None, focal_s=0.0, pool=None):
    """Reported departure (steps, float) for each vehicle of `day`.

    mode: 'M0' | 'M2' (w hours) | 'M3' (w hours, share q) | 'M4' (share rho, shift s hours) |
          'M1r' SUPPLEMENTARY (not in the submitted design): the driver-entered departure recorded in ACN-Data where it exists,
          otherwise an error drawn from `pool` (see empirical_errors) added to the true departure.
    focal/focal_s: paired replay override, vehicle `focal` reports dep - focal_s hours (applied on top of `mode`).
    """
    n = len(day.arr)
    dep = day.dep.astype(float)
    d = base_draws(seed, day_index, n)
    if mode == "M0":
        rep = dep.copy()
    elif mode == "M2":
        rep = dep + d["eps"] * w / DELTA
    elif mode == "M3":
        rep = dep + np.where(d["sel"] < q, d["late"] * w / DELTA, 0.0)
    elif mode == "M4":
        rep = dep - np.where(d["sel"] < rho, s / DELTA, 0.0)
    elif mode == "M1r":
        imputed = dep + pool[np.minimum((d["imp"] * len(pool)).astype(int), len(pool) - 1)]
        rep = np.where(np.isnan(day.rep_rec), imputed, day.rep_rec)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    if focal is not None:
        rep[focal] = dep[focal] - focal_s / DELTA
    return np.maximum(rep, day.arr.astype(float))
