"""Live demo (about 1 minute): one real test day, every method, then a paired-replay cheat test. Needs results/runs/e1_m0/seed0/best.pt.

Usage: python scripts/demo.py [day_index=58] [--plot]
Prints a metric table for FCFS / EDF / LLF / MPC / OPT / PPOLag on one truthful day, then asks: does one driver gain by reporting 1 h early?
--plot saves results/demo_day.png (station power over the day, learned policy vs EDF vs the grid cap).
"""
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")
from safecharge.agents.ppo_lag import PPOLag  # noqa: E402
from safecharge.baselines import rule_action  # noqa: E402
from safecharge.config import get_days, load_config  # noqa: E402
from safecharge.data import DELTA, N_PORTS, P_PORT  # noqa: E402
from safecharge.env import ChargingEnv  # noqa: E402
from safecharge.metrics import lp_metrics, paired_gain, rollout  # noqa: E402
from safecharge.mpc import mpc_action  # noqa: E402
from safecharge.offline_lp import solve_offline  # noqa: E402
from safecharge.report_model import reports  # noqa: E402

idx = int(next((a for a in sys.argv[1:] if a.isdigit()), 20))
cfg = load_config("configs/e1.yaml")
base, der = load_config(cfg["base"]), load_config(cfg["derived"])
day = get_days(base, "test")[idx]
env = ChargingEnv(N_PORTS["caltech"], der["P_grid"], e_ref=der["e_ref"], reward_scale=der["reward_scale"])
agent = PPOLag(cfg["agent"], N_PORTS["caltech"], der["P_grid"], der["e_ref"], der["reward_scale"], (der["d1"], der["d2"]))
agent.load(torch.load("results/runs/e1_m0/seed0/best.pt", weights_only=False))
rep = reports(day, "M0", 12345, idx)                       # truthful reports

methods = {"FCFS": lambda e, o: rule_action(e, "FCFS"), "EDF": lambda e, o: rule_action(e, "EDF"), "LLF": lambda e, o: rule_action(e, "LLF"),
           "MPC": lambda e, o: mpc_action(e, der["offline_penalty_M"]), "PPOLag": agent.policy_fn()}
rows, power = [], {}
for name, fn in methods.items():
    m, e = rollout(env, day, rep, fn)
    rows.append(dict(method=name, **{k: m[k] for k in ("cost_per_kwh", "unmet_share", "high_power_ratio", "grid_viol", "peak_power")}))
    power[name] = e.power_log.copy()
opt = lp_metrics(day, solve_offline(day, der["P_grid"], P_PORT, der["offline_penalty_M"]), P_PORT, der["P_grid"])
rows.append(dict(method="OPT (hindsight)", **{k: opt[k] for k in ("cost_per_kwh", "unmet_share", "high_power_ratio", "grid_viol", "peak_power")}))
print(f"\nDay {day.date}: {len(day.arr)} vehicles, grid cap {der['P_grid']:.1f} kW\n")
print(pd.DataFrame(rows).round(4).to_string(index=False))

print("\nPaired replay: 3 vehicles with stays >= 2 h each report 1 h EARLIER (others truthful). Extra energy received (% of own request):")
rng = np.random.default_rng(0)
focal = rng.choice(np.flatnonzero((day.dep - day.arr) * DELTA >= 2.0), 3, replace=False)
for name in ("EDF", "LLF", "PPOLag"):
    g = [paired_gain(env, day, rep, methods[name], int(f), 1.0)[1] for f in focal]
    print(f"  {name:7s}", "  ".join(f"{x:+.2f}%" for x in g))

if "--plot" in sys.argv:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(96) * DELTA
    plt.figure(figsize=(8, 3))
    for n, c in (("EDF", "#0072B2"), ("PPOLag", "#D55E00")):
        plt.step(t, power[n], where="post", label=n, color=c)
    plt.axhline(der["P_grid"], color="k", ls="--", lw=1, label="grid cap")
    plt.xlabel("hour of day"); plt.ylabel("station power (kW)"); plt.legend(frameon=False); plt.tight_layout()
    plt.savefig("results/demo_day.png", dpi=150)
    print("saved results/demo_day.png")
