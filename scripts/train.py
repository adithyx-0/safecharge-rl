"""Train PPO-Lagrangian. Usage: python scripts/train.py configs/e1.yaml --seed 0 [--updates N] [--out results/runs]
Resumable: re-running with the same config and seed continues from <out>/<name>/seed<k>/last.pt. Logs log.csv (one row per update,
validation columns every eval_every updates). best.pt = best validation checkpoint (feasible first, then lowest cost per kWh).
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, "src")
from safecharge.agents.ppo_lag import PPOLag  # noqa: E402
from safecharge.config import get_days, load_config  # noqa: E402
from safecharge.data import N_PORTS  # noqa: E402
from safecharge.report_model import empirical_errors  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--updates", type=int, default=None)
ap.add_argument("--out", default="results/runs")
ap.add_argument("--threads", type=int, default=1)
args = ap.parse_args()
torch.set_num_threads(args.threads)

cfg = load_config(args.config)
base, der = load_config(cfg["base"]), load_config(cfg["derived"])
site = base["site"]
train, val = get_days(base, "train"), get_days(base, "val")
n_updates = args.updates or cfg["train"]["n_updates"]
run = os.path.join(args.out, cfg["name"], f"seed{args.seed}")
os.makedirs(run, exist_ok=True)

agent = PPOLag(cfg["agent"], N_PORTS[site], der["P_grid"], der["e_ref"], der["reward_scale"], (der["d1"], der["d2"]), seed=args.seed)
agent.pool = empirical_errors(train)
last, best = os.path.join(run, "last.pt"), os.path.join(run, "best.pt")
bc_cfg = cfg["agent"].get("bc")
log_path = os.path.join(run, "log.csv")
rows, best_key = [], (9, 1e9)
if os.path.exists(last):
    agent.load(torch.load(last, weights_only=False))
    rows = pd.read_csv(log_path).to_dict("records") if os.path.exists(log_path) else []
    best_key = tuple(torch.load(best, weights_only=False)["key"]) if os.path.exists(best) else best_key
    print(f"resumed at update {agent.n_updates}")
elif bc_cfg:
    print(f"behaviour-cloning warm start from {bc_cfg['expert']}: final loss {agent.bc_pretrain(train, **bc_cfg):.4f}")


def val_summary():
    m = agent.evaluate(val)
    J1, J2 = np.mean([r["cost1_sum"] for r in m]), np.mean([r["cost2_sum"] for r in m])
    viol = max(0.0, J1 / der["d1"] - 1) + max(0.0, J2 / der["d2"] - 1)
    return dict(val_U=np.mean([r["unmet_share"] for r in m]), val_cost_per_kwh=np.nanmean([r["cost_per_kwh"] for r in m]),
                val_J1=J1, val_J2=J2, val_viol=viol, val_grid_viol=sum(r["grid_viol"] for r in m)), (0 if viol == 0 else 1, viol if viol else np.nanmean([r["cost_per_kwh"] for r in m]))


t0 = time.time()
while agent.n_updates < n_updates:
    agent.set_lr(agent.n_updates / n_updates)
    batch = agent.collect(train)
    row = dict(update=agent.n_updates + 1, **agent.update(batch))
    if agent.n_updates % cfg["train"]["eval_every"] == 0 or agent.n_updates == n_updates:
        vs, key = val_summary()
        row.update(vs)
        if key < best_key:
            best_key = key
            torch.save(dict(**agent.state(), key=key), best)
        torch.save(agent.state(), last)
        print(f"[{agent.n_updates:4d}] ret={row['ep_return']:7.2f} J1={row['J1']:.3f}/{der['d1']:.2f} J2={row['J2']:.2f}/{der['d2']:.2f} "
              f"lam=({row['lam1']:.2f},{row['lam2']:.2f}) | val U={vs['val_U']:.4f} $/kWh={vs['val_cost_per_kwh']:.4f} "
              f"J1={vs['val_J1']:.3f} J2={vs['val_J2']:.2f} gridviol={vs['val_grid_viol']} | {time.time()-t0:.0f}s", flush=True)
    rows.append(row)
    pd.DataFrame(rows).to_csv(log_path, index=False)
torch.save(agent.state(), last)
with open(os.path.join(run, "config_used.yaml"), "w") as f:
    yaml.safe_dump(dict(cfg=cfg, base=base, derived=der, seed=args.seed), f)
print("done", run)
