"""Descriptive check: how far are recorded driver-entered departures (userInputs.requestedDeparture) from the real
disconnect time? Error = requested - actual disconnect, in hours (negative = driver said earlier than they left).
Uses the LAST userInputs entry of each session. Writes results/recorded_report_error_<site>.csv. No design item is changed.
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from safecharge.data import load_sessions  # noqa: E402

rows = []
for site in ("caltech", "jpl"):
    df = load_sessions(f"data/raw/{site}_sessions.json")

    def last_req(ui):
        if isinstance(ui, list):
            for e in reversed(ui):
                if isinstance(e, dict) and e.get("requestedDeparture"):
                    return e["requestedDeparture"]
        return None

    req = df["userInputs"].apply(last_req)
    d = df[req.notna()].copy()
    d["req"] = pd.to_datetime(req[req.notna()], utc=True, format="mixed")
    d["err_h"] = (d["req"] - d["disc_utc"]).dt.total_seconds() / 3600
    d["stay_h"] = (d["disc_utc"] - d["conn_utc"]).dt.total_seconds() / 3600
    d["req_before_conn"] = d["req"] < d["conn_utc"]
    d["q"] = d["conn"].dt.to_period("Q").astype(str)
    print(f"\n== {site}: {len(d)} of {len(df)} sessions with a recorded departure ==")
    print(f"requested before connection time (unusable): {d.req_before_conn.mean():.2%}")
    g = d[~d.req_before_conn & d.err_h.between(-48, 48)]
    print(f"kept for error stats (requested >= connection, |err| <= 48 h): {len(g)} ({len(g)/len(d):.1%})")
    qs = g.err_h.quantile([.05, .25, .5, .75, .95]).round(2).to_dict()
    print(f"error hours  mean={g.err_h.mean():.2f}  mean|err|={g.err_h.abs().mean():.2f}  quantiles={qs}")
    print(f"share that said EARLIER than actual: {(g.err_h < 0).mean():.1%}; LATER (vehicle left earlier than announced): {(g.err_h > 0).mean():.1%}")
    for thr in (0.25, 0.5, 1, 2):
        print(f"   |err| <= {thr:>4} h: {(g.err_h.abs() <= thr).mean():.1%}   later by > {thr} h: {(g.err_h > thr).mean():.1%}   earlier by > {thr} h: {(g.err_h < -thr).mean():.1%}")
    t = g.groupby("q").agg(n=("err_h", "size"), mean_err_h=("err_h", "mean"), mean_abs_err_h=("err_h", lambda x: x.abs().mean()),
                          median_err_h=("err_h", "median"), p_later_1h=("err_h", lambda x: (x > 1).mean()), p_earlier_1h=("err_h", lambda x: (x < -1).mean())).round(3)
    t.to_csv(f"results/recorded_report_error_{site}.csv")
    print(t.to_string())
