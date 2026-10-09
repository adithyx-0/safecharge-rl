"""Per-day metrics (XI.A) and a generic rollout. Aggregate over days with `aggregate`, bootstrap CI over days."""
import time

import numpy as np

from .data import DELTA, T

DUR_BINS = [0, 2, 4, 8, 24.01]  # hours of stay; PROVISIONAL bins for the short-vs-long stratification


def rollout(env, day, rep, act_fn):
    """Run one day. act_fn(env, obs) -> u. Returns (metrics dict, env)."""
    obs, _ = env.reset(options=dict(day=day, rep=rep))
    done, c1, c2, steps, t0 = False, 0.0, 0.0, 0, time.perf_counter()
    while not done:
        u = act_fn(env, obs)
        obs, _, done, _, info = env.step(u)
        c1 += info["cost1"]
        c2 += info["cost2"]
        steps += 1
    ms = (time.perf_counter() - t0) / steps * 1000
    return day_metrics(env, c1, c2, ms), env


def _strat(d, E):
    """Unmet-energy share by length of stay (short vs long), to check short-stay vehicles are not quietly neglected."""
    e, dur, out = d.energy, (d.dep - d.arr) * DELTA, {}
    for lo, hi in zip(DUR_BINS[:-1], DUR_BINS[1:]):
        sel = (dur >= lo) & (dur < hi)
        out[f"U_dur_{lo:g}_{min(hi, 24):g}h"] = float(np.maximum(e[sel] - E[sel], 0).sum() / e[sel].sum()) if sel.any() else np.nan
    return out


def day_metrics(env, cost1_sum=np.nan, cost2_sum=np.nan, ms_per_step=np.nan):
    d = env.day
    e = d.energy
    n = len(e)
    E = env.E
    kwh = float(E.sum())
    dollars = float(np.sum(d.price * env.power_log * DELTA))
    s = E / e
    log = np.array(env.port_power_log).reshape(-1, 3)
    active = len(log)
    high = int(np.sum(log[:, 2] > 0.8 * env.P_port)) if active else 0
    strat = _strat(d, E)
    out = dict(
        date=str(d.date), n_veh=n, dropped=int(env.dropped.sum()),
        cost=dollars, kwh=kwh, cost_per_kwh=dollars / kwh if kwh > 0 else np.nan,
        unmet_share=float(np.maximum(e - E, 0).sum() / e.sum()),
        miss_rate=float(np.mean(E < 0.95 * e)),
        grid_viol=int(np.sum(env.power_log > env.P_grid + 1e-9)),
        grid_max_excess=float(max(0.0, np.max(env.power_log - env.P_grid))),
        high_power_ratio=high / active if active else np.nan,
        jain=float(s.sum() ** 2 / (n * np.sum(s ** 2))) if np.sum(s ** 2) > 0 else np.nan,
        cost1_sum=cost1_sum, cost2_sum=cost2_sum, ms_per_step=ms_per_step,
        peak_power=float(env.power_log.max()),
    )
    out.update(strat)
    return out


def bootstrap_ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n_boot, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_gain(env, day, base_rep, act_fn, focal, s):
    """Misreporting gain for vehicle `focal`: same day, same policy, same other reports; honest vs reporting dep - s hours.
    Returns (gain_kwh, gain_pct_of_e). act_fn must be deterministic (or reseeded identically) for a clean pairing."""
    rep_honest = base_rep.copy()
    rep_honest[focal] = max(float(day.dep[focal]), float(day.arr[focal]))
    rep_lie = base_rep.copy()
    rep_lie[focal] = max(float(day.dep[focal]) - s / DELTA, float(day.arr[focal]))
    _, e1 = rollout(env, day, rep_honest, act_fn)
    E_honest = e1.E[focal]
    _, e2 = rollout(env, day, rep_lie, act_fn)
    g = e2.E[focal] - E_honest
    return float(g), float(100 * g / day.energy[focal])


def lp_metrics(day, lp, P_port, P_grid):
    """Same metric columns as day_metrics, computed from an offline-LP solution (full information, no env)."""
    e = day.energy
    n = len(e)
    E = np.zeros(n)
    np.add.at(E, lp["idx"][lp["veh"]], lp["x"] * DELTA)
    s = E / e
    active = lp["x"] > 1e-9
    log_p = lp["x"][active]
    power = np.zeros(T)
    np.add.at(power, lp["t"], lp["x"])
    kwh = float(E.sum())
    return dict(date=str(day.date), n_veh=n, dropped=0, cost=lp["cost"], kwh=kwh, cost_per_kwh=lp["cost"] / kwh if kwh > 0 else np.nan,
                unmet_share=float(np.maximum(e - E, 0).sum() / e.sum()), miss_rate=float(np.mean(E < 0.95 * e - 1e-9)),
                grid_viol=int(np.sum(power > P_grid + 1e-6)), grid_max_excess=float(max(0.0, power.max() - P_grid)),
                high_power_ratio=float(np.mean(log_p > 0.8 * P_port)) if active.any() else np.nan,
                jain=float(s.sum() ** 2 / (n * np.sum(s ** 2))) if np.sum(s ** 2) > 0 else np.nan,
                cost1_sum=np.nan, cost2_sum=np.nan, ms_per_step=np.nan, peak_power=float(power.max()), **_strat(day, E))
