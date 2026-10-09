"""Evaluate methods on one split under one report mode with identical days and report draws. Same column set for every table.

Usage: python scripts/run_experiment.py configs/e1.yaml --split test --mode M0 [--w .. --q .. --rho .. --s ..] [--kappa 0.2] [--site caltech]
                                          [--tag name] [--seeds 0,1,2] [--ckpt best|last]
Methods: FCFS, EDF, LLF (rules, see reported departure only), MPC (receding-horizon LP on declared deadlines), OPT (offline LP, true departures), PPOLag (mean over the trained seeds found
under results/runs/<name>/seed*/; each seed evaluated separately, then averaged per test day). Output: results/<tag>_per_day.csv and
results/<tag>_summary.csv (mean and 95% bootstrap CI over test days; gap to optimum per day; extra column n_seeds).
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")
from safecharge.agents.ppo_lag import PPOLag  # noqa: E402
from safecharge.baselines import rule_action  # noqa: E402
from safecharge.config import get_days, load_config  # noqa: E402
from safecharge.data import N_PORTS, P_PORT  # noqa: E402
from safecharge.env import ChargingEnv  # noqa: E402
from safecharge.metrics import bootstrap_ci, lp_metrics, rollout  # noqa: E402
from safecharge.mpc import mpc_action  # noqa: E402
from safecharge.offline_lp import solve_offline  # noqa: E402
from safecharge.report_model import empirical_errors, reports  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--split", default="test")
ap.add_argument("--mode", default="M0")
for k in ("w", "q", "rho", "s"):
    ap.add_argument(f"--{k}", type=float, default=0.0)
ap.add_argument("--kappa", type=float, default=None)
ap.add_argument("--curtail", action="store_true", help="E4: evaluate in the curtail-at-declared-departure environment")
ap.add_argument("--site", default=None)
ap.add_argument("--tag", default=None)
ap.add_argument("--seeds", default=None)
ap.add_argument("--ckpt", default="best")
ap.add_argument("--runs", default="results/runs")
ap.add_argument("--report-seed", type=int, default=12345)
a = ap.parse_args()

cfg = load_config(a.config)
base, der = load_config(cfg["base"]), load_config(cfg["derived"])
site = a.site or base["site"]
train_pool = empirical_errors(get_days(base, "train"))        # M1r imputation pool always comes from the Caltech TRAIN days
days = get_days(base, a.split, site)
P_peak = der["P_peak"]
if site != base["site"]:  # transfer site: same kappa rule, P_peak from THAT site's train days (median daily full-rate-on-arrival peak)
    from safecharge.data import P_PORT as _PP, T as _T
    def _peak(d):
        x = np.zeros(_T)
        for lo, hi in zip(d.arr, d.dep):
            x[lo:hi] += _PP
        return x.max()
    P_peak = float(np.median([_peak(d) for d in get_days(base, "train", site)]))
P_grid = (a.kappa or der["kappa"]) * P_peak
N, M = N_PORTS[site], der["offline_penalty_M"]
kw = {k: getattr(a, k) for k in ("w", "q", "rho", "s") if getattr(a, k)}
tag = a.tag or f"{cfg['name']}_{site}_{a.split}_{a.mode}" + "".join(f"_{k}{v:g}" for k, v in kw.items()) + (f"_k{a.kappa:g}" if a.kappa else "") + ("_curtail" if a.curtail else "")
reps = [reports(d, a.mode, a.report_seed, i, pool=train_pool, **kw) for i, d in enumerate(days)]  # same draws for every method
env = ChargingEnv(N, P_grid, e_ref=der["e_ref"], reward_scale=der["reward_scale"], curtail=a.curtail)

per = {}
for kind in ("FCFS", "EDF", "LLF"):
    per[kind] = [rollout(env, d, r, lambda e, o, k=kind: rule_action(e, k))[0] for d, r in zip(days, reps)]
per["MPC"] = [rollout(env, d, r, lambda e, o: mpc_action(e, der["offline_penalty_M"]))[0] for d, r in zip(days, reps)]
lps = [solve_offline(d, P_grid, P_PORT, M) for d in days]
per["OPT"] = [lp_metrics(d, lp, P_PORT, P_grid) for d, lp in zip(days, lps)]
for d, lp, r in zip(days, lps, reps):                           # lower-bound check on every day (objective = cost + M * unmet)
    for kind in ("FCFS", "EDF", "LLF", "MPC"):
        m = next(x for x in per[kind] if x["date"] == str(d.date))
        assert lp["objective"] <= m["cost"] + M * m["unmet_share"] * d.energy.sum() + 1e-6, ("LP bound violated", kind, d.date)

seed_dirs = sorted(glob.glob(os.path.join(a.runs, cfg["name"], "seed*")))
if a.seeds:
    seed_dirs = [os.path.join(a.runs, cfg["name"], f"seed{s}") for s in a.seeds.split(",")]
rl = []
for sd in seed_dirs:
    f = os.path.join(sd, f"{a.ckpt}.pt")
    if not os.path.exists(f):
        continue
    ag = PPOLag(cfg["agent"], N, P_grid, der["e_ref"], der["reward_scale"], (der["d1"], der["d2"]))
    ag.load(torch.load(f, weights_only=False))
    ag.env_kw["N"] = N
    rl.append([rollout(env, d, r, ag.policy_fn())[0] for d, r in zip(days, reps)])
    # lower bound check for RL too
    for d, lp, m in zip(days, lps, rl[-1]):
        assert lp["objective"] <= m["cost"] + M * m["unmet_share"] * d.energy.sum() + 1e-6, ("LP bound violated", "RL", d.date)
if rl:
    per["PPOLag"] = [{k: (np.mean([r[i][k] for r in rl]) if isinstance(rl[0][i][k], (int, float, np.floating)) else rl[0][i][k]) for k in rl[0][i]} for i in range(len(days))]

cols = ["cost", "cost_per_kwh", "unmet_share", "miss_rate", "grid_viol", "grid_max_excess", "high_power_ratio", "jain", "cost1_sum", "cost2_sum", "ms_per_step", "peak_power"]
strat = [c for c in per["FCFS"][0] if c.startswith("U_dur_")]
opt = pd.DataFrame(per["OPT"])
allrows, summ = [], []
for meth, rows in per.items():
    df = pd.DataFrame(rows)
    df.insert(0, "method", meth)
    df["gap_cost_per_kwh"] = (df["cost_per_kwh"] - opt["cost_per_kwh"]) / opt["cost_per_kwh"]
    df["gap_U"] = df["unmet_share"] - opt["unmet_share"]
    allrows.append(df)
    srow = dict(method=meth, n_days=len(df), n_seeds=len(rl) if meth == "PPOLag" else 1)
    for c in cols + strat + ["gap_cost_per_kwh", "gap_U"]:
        m, lo, hi = bootstrap_ci(df[c].to_numpy(float))
        srow[c], srow[c + "_lo"], srow[c + "_hi"] = m, lo, hi
    if meth == "PPOLag" and len(rl) > 1:
        srow["U_seed_std"] = float(np.std([np.mean([x["unmet_share"] for x in r]) for r in rl]))
        srow["cost_per_kwh_seed_std"] = float(np.nanstd([np.nanmean([x["cost_per_kwh"] for x in r]) for r in rl]))
    summ.append(srow)
os.makedirs("results", exist_ok=True)
pd.concat(allrows).to_csv(f"results/{tag}_per_day.csv", index=False)
S = pd.DataFrame(summ)
S.to_csv(f"results/{tag}_summary.csv", index=False)
pd.set_option("display.width", 250)
print(f"\n{tag}: site={site} split={a.split} mode={a.mode} {kw} P_grid={P_grid:.1f} kW, {len(days)} days, {len(rl)} RL seeds")
print(S[["method", "cost_per_kwh", "gap_cost_per_kwh", "unmet_share", "miss_rate", "grid_viol", "high_power_ratio", "jain", "cost2_sum", "ms_per_step"]].round(4).to_string(index=False))
