"""kappa for the transfer site by the SAME rule as the main site: tightest kappa where the offline optimum's unmet share U* is still <= 0.6%
(Caltech: kappa 0.20 gives U* = 0.56%). Uses that site's TRAIN days, truthful reports. Writes results/calibration_kappa_<site>.csv.
Usage: python scripts/calibrate_transfer_site.py [site=jpl]
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from safecharge.config import get_days, load_config  # noqa: E402
from safecharge.data import P_PORT, T  # noqa: E402
from safecharge.offline_lp import solve_offline  # noqa: E402

site = sys.argv[1] if len(sys.argv) > 1 else "jpl"
base = load_config("configs/base.yaml")
days = get_days(base, "train", site)


def peak(d):
    x = np.zeros(T)
    for a, b in zip(d.arr, d.dep):
        x[a:b] += P_PORT
    return x.max()


P_peak = float(np.median([peak(d) for d in days]))
rows = []
for k in (0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
    u = np.mean([solve_offline(d, k * P_peak, P_PORT, 5.0)["unmet"] / d.energy.sum() for d in days])
    rows.append(dict(site=site, kappa=k, P_grid=round(k * P_peak, 1), OPT_U=round(float(u), 4)))
tab = pd.DataFrame(rows)
tab.to_csv(f"results/calibration_kappa_{site}.csv", index=False)
print(f"{site}: {len(days)} train days, P_peak = {P_peak:.1f} kW, mean vehicles/day {np.mean([len(d.arr) for d in days]):.1f}, mean kWh/day {np.mean([d.energy.sum() for d in days]):.0f}")
print(tab.to_string(index=False))
ok = tab[tab.OPT_U <= 0.006]
print("chosen kappa (tightest with U* <= 0.6%):", ok.kappa.min() if len(ok) else "none in grid")
