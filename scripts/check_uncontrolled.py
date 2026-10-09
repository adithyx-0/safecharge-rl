"""Uncontrolled replay check: every vehicle at full rate on arrival, no grid cap. Reports the aggregate energy
shortfall versus recorded kWhDelivered caused by 15-min rounding and clipping e_i to P_port * window.

Usage: python scripts/check_uncontrolled.py [site] [path] [midnight: drop|clip]
"""
import sys

import numpy as np

sys.path.insert(0, "src")
from safecharge.data import N_PORTS, build_days, load_sessions  # noqa: E402
from safecharge.env import ChargingEnv  # noqa: E402
from safecharge.metrics import rollout  # noqa: E402

site = sys.argv[1] if len(sys.argv) > 1 else "caltech"
path = sys.argv[2] if len(sys.argv) > 2 else f"data/raw/{site}_sessions.json"
mid = sys.argv[3] if len(sys.argv) > 3 else "drop"
df = load_sessions(path)
days, st = build_days(df, midnight=mid)
env = ChargingEnv(N_PORTS[site], 1e9)
sim = rec = 0.0
dropped = 0
worst = 0.0
for d in days:
    m, env = rollout(env, d, d.dep.astype(float), lambda e, o: np.ones(e.N))
    sim += env.E.sum()
    rec += d.kwh_rec.sum()
    dropped += int(env.dropped.sum())
    worst = max(worst, float(np.max(np.abs(env.E - d.energy))))
print(f"{site} midnight={mid}: days={len(days)} simulated={sim:.1f} kWh recorded(same sessions)={rec:.1f} kWh")
print(f"shortfall = {rec - sim:.1f} kWh = {(rec - sim) / rec:.2%} of recorded; vehicles with no free port: {dropped}")
print(f"max |E_i - clipped e_i| over all vehicles = {worst:.2e} (should be ~0: uncontrolled replay delivers the clipped request)")
