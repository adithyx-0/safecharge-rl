"""Tiny yaml config + split loader. Day lists are cached under data/processed (gitignored)."""
import os
import pickle

import pandas as pd
import yaml

from .data import build_days, load_sessions


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_days(cfg, split, site=None, raw_dir="data/raw", cache_dir="data/processed"):
    """Days of `split` ('train'|'val'|'test'|'post') for `site` (default cfg['site']), already filtered per cfg."""
    site = site or cfg["site"]
    key = f"{site}_{cfg['midnight']}_{cfg['tariff']}_{'-'.join(cfg['exclude_stations'].get(site, []))}.pkl"
    path = os.path.join(cache_dir, key)
    if os.path.exists(path):
        with open(path, "rb") as f:
            days = pickle.load(f)
    else:
        df = load_sessions(os.path.join(raw_dir, f"{site}_sessions.json"))
        days, _ = build_days(df, midnight=cfg["midnight"], tariff=cfg["tariff"],
                             exclude_stations=cfg["exclude_stations"].get(site, []))
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(days, f)
    a, b = (pd.Timestamp(x).date() for x in cfg["splits"][split])
    return [d for d in days if a <= d.date <= b]
