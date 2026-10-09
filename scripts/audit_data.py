"""Phase 0 data audit. Prints tables; the user picks split dates from them. No decisions are made here.

Usage: python scripts/audit_data.py [site] [path]     (defaults: caltech data/raw/caltech_sessions.json)
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from safecharge.data import DELTA, N_PORTS, P_PORT, T, build_days, load_sessions, peak_concurrency  # noqa: E402

site = sys.argv[1] if len(sys.argv) > 1 else "caltech"
path = sys.argv[2] if len(sys.argv) > 2 else f"data/raw/{site}_sessions.json"
pd.set_option("display.width", 200)

df = load_sessions(path)
print(f"\n== {site}: {len(df)} sessions, {df.conn.min()} .. {df.conn.max()} (local) ==")
print("duplicate sessionIDs:", int(df.sessionID.duplicated().sum()))
ui = df["userInputs"].apply(lambda x: bool(x))
print(f"userInputs populated: {ui.mean():.3%}")
rd = df["userInputs"].apply(lambda x: any(isinstance(e, dict) and e.get("requestedDeparture") for e in x) if isinstance(x, list) else False)
print(f"userInputs.requestedDeparture populated: {rd.mean():.3%}")

df["date"] = df.conn.dt.normalize()
wk = df.date.dt.dayofweek < 5
df["ym"] = df.conn.dt.to_period("M")
tab = pd.DataFrame({
    "sessions": df.groupby("ym").size(),
    "weekday_sessions": df[wk].groupby("ym").size(),
    "weekdays_with_data": df[wk].groupby("ym").date.nunique(),
    "zero_kwh": df[df.kwh <= 0].groupby("ym").size(),
}).fillna(0).astype(int)
print("\nSessions per month (local connection time):")
print(tab.to_string())

print("\nOption comparison (tariff day counts and exclusions), after dropping zero-energy/invalid sessions:")
rows = []
for mid in ("drop", "clip"):
    days, st = build_days(df, midnight=mid)
    ds = pd.Series([d.date for d in days])
    summer = int(sum(1 for d in ds if 6 <= d.month <= 9))
    cc = np.array([peak_concurrency(d) for d in days])
    rows.append(dict(midnight=mid, weekdays=len(days), summer_weekdays=summer, dst_days_dropped=st["dst_day"],
                     crosses_midnight=st["crosses_midnight"], sessions_in_days=int(sum(len(d.arr) for d in days)),
                     days_over_N_ports=int((cc > N_PORTS[site]).sum()), max_concurrent=int(cc.max()),
                     energy_clipped=st["energy_clipped"], kwh_recorded=round(st["kwh_recorded"]),
                     kwh_after_clip=round(st["kwh_after_clip"])))
print(pd.DataFrame(rows).to_string(index=False))
print(f"\nzero-energy sessions: {(df.kwh <= 0).sum()}; invalid range (disc <= conn): {(df.disc <= df.conn).sum()}")
print(f"midnight-crossing share of all sessions: {(df.disc.dt.normalize() > df.date).mean():.2%}")

days, st = build_days(df, midnight="drop")
print("\nWeekday count per year (midnight='drop'):")
print(pd.Series([d.date.year for d in days]).value_counts().sort_index().to_string())
