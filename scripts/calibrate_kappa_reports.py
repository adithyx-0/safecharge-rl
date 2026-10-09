"""Where does the grid cap make reports matter? For kappa in a grid, unmet-energy share U (mean over TRAIN days) of the offline optimum
(true departures) and of EDF / LLF under M0, M2(w=2h), M3(q=.5,w=2h), M4(rho=.25,s=1h), M1r (recorded). Writes results/calibration_kappa_reports.csv.
Heuristics see only reported departures; the vehicle always leaves at its true departure.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from safecharge.baselines import rule_action  # noqa: E402
from safecharge.config import get_days, load_config  # noqa: E402
from safecharge.data import N_PORTS, P_PORT, T  # noqa: E402
from safecharge.env import ChargingEnv  # noqa: E402
from safecharge.metrics import rollout  # noqa: E402
from safecharge.offline_lp import solve_offline  # noqa: E402
from safecharge.report_model import empirical_errors, reports  # noqa: E402

cfg = load_config("configs/base.yaml")
days = get_days(cfg, "train")
pool = empirical_errors(days)
site = cfg["site"]
peak = np.median([np.max(np.bincount(np.concatenate([np.arange(a, b) for a, b in zip(d.arr, d.dep)]), minlength=T)) * P_PORT for d in days])
MODES = dict(M0={}, M2=dict(w=2.0), M3=dict(w=2.0, q=0.5), M4=dict(rho=0.25, s=1.0), M1r=dict(pool=pool))
rows = []
for kappa in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
    Pg = kappa * peak
    env = ChargingEnv(N_PORTS[site], Pg)
    row = dict(kappa=kappa, P_grid=round(Pg, 1))
    row["OPT_U"] = round(float(np.mean([solve_offline(d, Pg, P_PORT, 5.0)["unmet"] / d.energy.sum() for d in days])), 4)
    for kind in ("EDF", "LLF"):
        for mode, kw in MODES.items():
            u = [rollout(env, d, reports(d, mode, 0, i, **kw), lambda e, o, k=kind: rule_action(e, k))[0]["unmet_share"] for i, d in enumerate(days)]
            row[f"{kind}_{mode}"] = round(float(np.mean(u)), 4)
    rows.append(row)
    print(row, flush=True)
tab = pd.DataFrame(rows)
tab.to_csv("results/calibration_kappa_reports.csv", index=False)
pd.set_option("display.width", 250)
print(tab.to_string(index=False))
