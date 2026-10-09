"""Load ACN-Data sessions and turn them into per-day arrays on the 15-min grid.

Open decisions from CLAUDE.md are options here, with PROVISIONAL defaults (the user decides):
  midnight: 'drop' | 'clip'      sessions that cross local midnight
  rounding: arrival rounded up, departure rounded down (conservative), e_i clipped to P_port * window.
Days are bucketed in local time (session `timezone`); DST-change days (92/100 steps) are dropped.
"""
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

T = 96
DELTA = 0.25  # hours
P_PORT = 6.656  # kW
N_PORTS = {"caltech": 54, "jpl": 52}

# SCE TOU-EV-4 (March 2019), weekday periods start at hours [0, 8, 12, 18, 23]
_HOURS = [0, 8, 12, 18, 23]
_SUMMER = [0.05623, 0.0925, 0.26668, 0.0925, 0.05623]
_WINTER = [0.06087, 0.07492, 0.0869, 0.07492, 0.06087]


def price_curve(date, tariff="real"):
    """Price per step ($/kWh) for a weekday. tariff: 'real' (summer Jun-Sep, else winter) | 'summer' (always summer)."""
    summer = tariff == "summer" or (tariff == "real" and 6 <= date.month <= 9)
    vals = _SUMMER if summer else _WINTER
    hour = np.arange(T) * DELTA
    idx = np.searchsorted(_HOURS, hour, side="right") - 1
    return np.asarray(vals)[idx]


@dataclass
class Day:
    date: object
    arr: np.ndarray     # first step present
    dep: np.ndarray     # first step gone (vehicle charges in steps arr..dep-1)
    energy: np.ndarray  # e_i kWh (after clipping)
    price: np.ndarray   # (T,)
    session_ids: list
    kwh_rec: np.ndarray = None  # recorded kWhDelivered before clipping to the window
    rep_rec: np.ndarray = None  # driver-entered departure in steps from local midnight (float), NaN if none recorded


def _last_requested(ui):
    if isinstance(ui, list):
        for e in reversed(ui):
            if isinstance(e, dict) and e.get("requestedDeparture"):
                return e["requestedDeparture"]
    return None


def _recorded_departure(df, tz):
    """Driver-entered requestedDeparture (last entry of userInputs) as local naive time, NaT if none."""
    utc = pd.to_datetime(df["userInputs"].apply(_last_requested), utc=True, format="mixed")
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for z, g in utc.groupby(tz):
        ok = g.notna()
        if ok.any():
            out.loc[g.index[ok]] = g[ok].dt.tz_convert(z).dt.tz_localize(None)
    return out


def load_sessions(path):
    """Raw JSON -> DataFrame with local arrival/departure timestamps and kWh."""
    with open(path) as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    df["conn_utc"] = pd.to_datetime(df["connectionTime"], utc=True, format="mixed")
    df["disc_utc"] = pd.to_datetime(df["disconnectTime"], utc=True, format="mixed")
    tz = df["timezone"].fillna("America/Los_Angeles")
    conn, disc = [], []
    for z, g in df.groupby(tz):
        conn.append(g["conn_utc"].dt.tz_convert(z).dt.tz_localize(None))
        disc.append(g["disc_utc"].dt.tz_convert(z).dt.tz_localize(None))
    df["conn"] = pd.concat(conn).reindex(df.index)
    df["disc"] = pd.concat(disc).reindex(df.index)
    df["req"] = _recorded_departure(df, tz)
    df["kwh"] = df["kWhDelivered"].astype(float)
    df["tz"] = tz
    return df


def is_dst_day(date, tz):
    d0 = pd.Timestamp(date).tz_localize(tz)
    d1 = (pd.Timestamp(date) + pd.Timedelta(days=1)).tz_localize(tz)
    return d1 - d0 != pd.Timedelta(hours=24)


def build_days(df, midnight="drop", tariff="real", weekdays_only=True, exclude_stations=()):
    """Return (list[Day], stats dict). Stats count every exclusion so the share affected can be reported."""
    if len(exclude_stations):
        df = df[~df["stationID"].isin(list(exclude_stations))]
    stats = dict(total=len(df), zero_energy=0, invalid_range=0, crosses_midnight=0, dst_day=0, empty_window=0,
                 energy_clipped=0, kwh_recorded=0.0, kwh_after_clip=0.0)
    d = df[(df["kwh"] > 0)]
    stats["zero_energy"] = len(df) - len(d)
    ok = d["disc"] > d["conn"]
    stats["invalid_range"] = int((~ok).sum())
    d = d[ok].copy()
    d["date"] = d["conn"].dt.normalize()
    cross = d["disc"].dt.normalize() > d["date"]
    stats["crosses_midnight"] = int(cross.sum())
    if midnight == "drop":
        d = d[~cross]
    d["a"] = np.ceil((d["conn"] - d["date"]) / pd.Timedelta(minutes=15)).astype(int)
    dep = np.floor((d["disc"] - d["date"]) / pd.Timedelta(minutes=15)).astype(int)
    d["b"] = np.minimum(dep, T)  # 'clip' mode: departure clipped to day end
    keep = d["b"] > d["a"]
    stats["empty_window"] = int((~keep).sum())
    d = d[keep]
    if weekdays_only:
        d = d[d["date"].dt.dayofweek < 5]
    cap = P_PORT * (d["b"] - d["a"]) * DELTA
    stats["energy_clipped"] = int((d["kwh"] > cap + 1e-9).sum())
    stats["kwh_recorded"] = float(d["kwh"].sum())
    d["rep_step"] = (d["req"] - d["date"]) / pd.Timedelta(minutes=15)
    d["e"] = np.minimum(d["kwh"], cap)
    stats["kwh_after_clip"] = float(d["e"].sum())
    days = []
    for date, g in d.groupby("date"):
        tz = g["tz"].iloc[0]
        if is_dst_day(date, tz):
            stats["dst_day"] += 1
            continue
        g = g.sort_values(["a", "b"])
        days.append(Day(date.date(), g["a"].to_numpy(), g["b"].to_numpy(), g["e"].to_numpy(),
                        price_curve(date, tariff), g["sessionID"].tolist(), g["kwh"].to_numpy(),
                        g["rep_step"].to_numpy(dtype=float)))
    return days, stats


def peak_concurrency(day):
    occ = np.zeros(T + 1, int)
    for a, b in zip(day.arr, day.dep):
        occ[a:b] += 1
    return int(occ.max())
