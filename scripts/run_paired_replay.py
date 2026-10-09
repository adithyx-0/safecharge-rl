"""E3 / E4: misreporting gain by paired replay. For sampled focal vehicles, run the SAME day twice with the same deterministic policy and the
same background reports; once the focal vehicle reports honestly (its true departure), once reporting dep - s hours. Gain = E_i(lie) - E_i(honest),
in kWh and % of e_i. Focal vehicles: stays >= 2 h (so every s <= 2 h is a real shift, not clipped at arrival), `--focal` random ones per day (seeded).
Methods: EDF, LLF, MPC, PPOLag (all seeds found; mean over seeds per focal vehicle). Mean over focal vehicles per day, then bootstrap CI over DAYS.
Usage: python scripts/run_paired_replay.py configs/e1.yaml --split test --bg M0 [--curtail] [--s 0.5,1,2] [--focal 5] [--tag name] [--runs results/runs]
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
from safecharge.data import DELTA, N_PORTS  # noqa: E402
from safecharge.env import ChargingEnv  # noqa: E402
from safecharge.metrics import bootstrap_ci, paired_gain  # noqa: E402
from safecharge.mpc import mpc_action  # noqa: E402
from safecharge.report_model import empirical_errors, reports  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--split", default="test")
ap.add_argument("--bg", default="M0", help="background report mode of the other drivers (M0, M2, M4, M1r)")
for k in ("w", "q", "rho", "s_bg"):
    ap.add_argument(f"--{k}", type=float, default=0.0)
ap.add_argument("--s", default="0.5,1,2")
ap.add_argument("--focal", type=int, default=5)
ap.add_argument("--curtail", action="store_true")
ap.add_argument("--kappa", type=float, default=None)
ap.add_argument("--tag", default=None)
ap.add_argument("--runs", default="results/runs")
ap.add_argument("--ckpt", default="best")
ap.add_argument("--site", default=None)
ap.add_argument("--max-days", type=int, default=None)
ap.add_argument("--no-rl", action="store_true")
ap.add_argument("--seeds", default=None, help="comma-separated RL seeds to use (default: all with a checkpoint)")
a = ap.parse_args()

cfg = load_config(a.config)
base, der = load_config(cfg["base"]), load_config(cfg["derived"])
site = a.site or base["site"]
days = get_days(base, a.split, site)[: a.max_days]
pool = empirical_errors(get_days(base, "train"))
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
env = ChargingEnv(N_PORTS[site], P_grid, e_ref=der["e_ref"], reward_scale=der["reward_scale"], curtail=a.curtail)
shifts = [float(x) for x in a.s.split(",")]
bg_kw = {k.replace("s_bg", "s"): getattr(a, k) for k in ("w", "q", "rho", "s_bg") if getattr(a, k)}
tag = a.tag or f"{cfg['name']}_paired_{a.split}_bg{a.bg}" + ("_curtail" if a.curtail else "")

methods = {"EDF": lambda e, o: rule_action(e, "EDF"), "LLF": lambda e, o: rule_action(e, "LLF"), "MPC": lambda e, o: mpc_action(e, der["offline_penalty_M"])}
agents = []
if not a.no_rl:
    sdirs = sorted(glob.glob(os.path.join(a.runs, cfg["name"], "seed*")))
    if a.seeds:
        sdirs = [os.path.join(a.runs, cfg["name"], f"seed{s}") for s in a.seeds.split(",")]
    for sd in sdirs:
        f = os.path.join(sd, f"{a.ckpt}.pt")
        if os.path.exists(f):
            ag = PPOLag(cfg["agent"], N_PORTS[site], P_grid, der["e_ref"], der["reward_scale"], (der["d1"], der["d2"]))
            ag.load(torch.load(f, weights_only=False))
            agents.append(ag)
    print(f"{len(agents)} RL seeds")

rng = np.random.default_rng(2024)                                # focal choice is the same for every method
rows = []
for i, d in enumerate(days):
    base_rep = reports(d, a.bg, 12345, i, pool=pool, **bg_kw)
    ok = np.flatnonzero((d.dep - d.arr) * DELTA >= 2.0)
    focal = rng.choice(ok, min(a.focal, len(ok)), replace=False) if len(ok) else []
    for f in focal:
        for s in shifts:
            for name, fn in methods.items():
                g, p = paired_gain(env, d, base_rep, fn, int(f), s)
                rows.append(dict(method=name, date=str(d.date), focal=int(f), s=s, gain_kwh=g, gain_pct=p, e_i=d.energy[f]))
            if agents:
                gp = [paired_gain(env, d, base_rep, ag.policy_fn(), int(f), s) for ag in agents]
                rows.append(dict(method="PPOLag", date=str(d.date), focal=int(f), s=s, gain_kwh=np.mean([x[0] for x in gp]),
                                 gain_pct=np.mean([x[1] for x in gp]), e_i=d.energy[f]))
os.makedirs("results", exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(f"results/{tag}_focal.csv", index=False)
summ = []
for (m, s), g in df.groupby(["method", "s"]):
    per_day = g.groupby("date")[["gain_kwh", "gain_pct"]].mean()
    r = dict(method=m, s=s, n_pairs=len(g), n_days=len(per_day), share_gain_gt0=float((g.gain_kwh > 1e-9).mean()), share_gain_lt0=float((g.gain_kwh < -1e-9).mean()))
    for c in ("gain_kwh", "gain_pct"):
        r[c], r[c + "_lo"], r[c + "_hi"] = bootstrap_ci(per_day[c].to_numpy())
    summ.append(r)
S = pd.DataFrame(summ)
S.to_csv(f"results/{tag}_summary.csv", index=False)
pd.set_option("display.width", 200)
print(f"\n{tag}: bg={a.bg} {bg_kw} curtail={a.curtail} P_grid={P_grid:.1f} kW, {len(days)} days")
print(S.round(3).to_string(index=False))
