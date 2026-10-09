"""Collect results/<prefix>_<E?>_<scenario>_summary.csv into uniform long tables and compact pivots. Usage: python scripts/make_tables.py [prefix=e1_m0]
Writes results/tables/<prefix>_<E>_long.csv (same columns as run_experiment summaries + scenario) and <prefix>_<E>_pivot_<metric>.csv.
E3 (paired replay) summaries have their own columns and are merged the same way.
"""
import glob
import os
import re
import sys

import pandas as pd

prefix = sys.argv[1] if len(sys.argv) > 1 else "e1_m0"
os.makedirs("results/tables", exist_ok=True)
pd.set_option("display.width", 250)
for exp in ("E1", "E2", "E3", "E6"):
    files = sorted(glob.glob(f"results/{prefix}_{exp}_*_summary.csv"))
    if not files:
        continue
    parts = []
    for f in files:
        d = pd.read_csv(f)
        d.insert(0, "scenario", re.sub(rf"^{prefix}_{exp}_|_summary\.csv$", "", os.path.basename(f)))
        parts.append(d)
    L = pd.concat(parts, ignore_index=True)
    L.to_csv(f"results/tables/{prefix}_{exp}_long.csv", index=False)
    metrics = ["gain_kwh", "gain_pct", "share_gain_gt0"] if exp == "E3" else ["unmet_share", "cost_per_kwh", "miss_rate", "high_power_ratio", "gap_cost_per_kwh"]
    idx = ["method", "s"] if exp == "E3" else ["method"]
    for m in metrics:
        if m in L:
            P = L.pivot_table(index=idx, columns="scenario", values=m)
            P.to_csv(f"results/tables/{prefix}_{exp}_pivot_{m}.csv")
    print(f"\n===== {exp}: {len(files)} scenarios =====")
    m0 = metrics[0]
    print(f"{m0} (rows: {idx}, columns: scenario)")
    print(L.pivot_table(index=idx, columns="scenario", values=m0).round(4).to_string())
    if exp != "E3":
        print("cost_per_kwh")
        print(L.pivot_table(index=idx, columns="scenario", values="cost_per_kwh").round(4).to_string())
