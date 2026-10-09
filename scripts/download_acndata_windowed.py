"""Download all ACN-Data sessions by time window (month by month), verifying each window against the API's own total.

Why: deep offset paging (`page=N` with N large) made the server return 502/timeouts, so the plain downloader skipped
records. Filtering by `connectionTime` keeps every query shallow. Each finished window is saved to
data/raw/windows/<site>/<start>.json, so re-running resumes. A window whose retrieved count differs from `_meta.total`,
or that fails, is split in half (down to one day). Anything still incomplete is listed at the end and the merged file is NOT written.

Token comes from the ACN_TOKEN env var (Bearer auth). Usage: python scripts/download_acndata_windowed.py [site ...]
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import requests

BASE = "https://ev.caltech.edu/api/v1/"
OUT = os.environ.get("ACN_OUT_DIR", "data/raw")
RETRIES, PAGE_DELAY = 6, 0.3
EXPECTED = {"caltech": 31424, "jpl": 33638}  # as of Sep 2026 (CLAUDE.md); the API total is the real check
HEAD = {"Authorization": f"Bearer {os.environ.get('ACN_TOKEN', '')}"}
failed = []


def http_date(dt):
    return format_datetime(dt, usegmt=True)


def get(url):
    for k in range(RETRIES):
        try:
            r = requests.get(url, headers=HEAD, timeout=45)
            if r.status_code == 200:
                return r.json()
            if r.status_code < 500:
                print(f"  HTTP {r.status_code}: {r.text[:200]}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"  {type(e).__name__}, retry {k + 1}/{RETRIES}")
        time.sleep(min(2 ** k, 30))
    return None


def fetch_window(site, a, b):
    """All sessions with a <= connectionTime < b, as (items, total) or None on failure."""
    where = f'connectionTime>="{http_date(a)}" and connectionTime<"{http_date(b)}"'
    url = BASE + f"sessions/{site}?max_results=100&where={where}"
    items, total = [], None
    while url:
        p = get(url)
        if p is None:
            return None
        total = p["_meta"]["total"]
        items.extend(p["_items"])
        href = p.get("_links", {}).get("next", {}).get("href")
        url = BASE + href if href else None
        time.sleep(PAGE_DELAY)
    return items, total


def do_window(site, a, b, d):
    path = os.path.join(d, a.strftime("%Y%m%d%H%M") + "_" + b.strftime("%Y%m%d%H%M") + ".json")
    if os.path.exists(path):
        return
    res = fetch_window(site, a, b)
    if res is not None and len(res[0]) == res[1]:
        with open(path, "w") as f:
            json.dump(res[0], f)
        print(f"{site} {a:%Y-%m-%d %H:%M} .. {b:%Y-%m-%d %H:%M}: {len(res[0])} sessions OK")
        return
    if b - a <= timedelta(days=1):
        got = None if res is None else f"{len(res[0])}/{res[1]}"
        failed.append((site, str(a), str(b), got))
        print(f"{site} {a:%Y-%m-%d}: FAILED ({got})")
        return
    mid = a + (b - a) / 2
    print(f"{site} {a:%Y-%m-%d} .. {b:%Y-%m-%d}: incomplete, splitting")
    do_window(site, a, mid, d)
    do_window(site, mid, b, d)


def edge(site, sort):
    p = get(BASE + f"sessions/{site}?max_results=1&sort={sort}connectionTime")
    return datetime.strptime(p["_items"][0]["connectionTime"], "%a, %d %b %Y %H:%M:%S GMT").replace(tzinfo=timezone.utc)


def run_site(site):
    d = os.path.join(OUT, "windows", site)
    os.makedirs(d, exist_ok=True)
    site_total = get(BASE + f"sessions/{site}?max_results=1")["_meta"]["total"]
    first, last = edge(site, ""), edge(site, "-")
    print(f"{site}: API total {site_total} (expected {EXPECTED[site]}), {first} .. {last}")
    a = first.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while a <= last:
        b = (a + timedelta(days=32)).replace(day=1)
        do_window(site, a, b, d)
        a = b
    items = {}
    for fn in sorted(os.listdir(d)):
        with open(os.path.join(d, fn)) as f:
            for s in json.load(f):
                items[s["_id"]] = s
    print(f"{site}: merged {len(items)} unique sessions, API total {site_total}")
    if len(items) != site_total or any(x[0] == site for x in failed):
        print(f"{site}: INCOMPLETE ({len(items)} of {site_total}); merged file NOT written. failed windows: {[x for x in failed if x[0] == site]}")
        return False
    out = os.path.join(OUT, f"{site}_sessions.json")
    with open(out, "w") as f:
        json.dump(sorted(items.values(), key=lambda s: s["connectionTime"] and datetime.strptime(s["connectionTime"], "%a, %d %b %Y %H:%M:%S GMT")), f)
    print(f"{site}: COMPLETE -> {out}")
    return True


if __name__ == "__main__":
    if not os.environ.get("ACN_TOKEN"):
        sys.exit("Set ACN_TOKEN first.")
    sites = sys.argv[1:] or ["caltech", "jpl"]
    ok = {s: run_site(s) for s in sites}
    print("\nSummary:", {s: ("complete" if v else "INCOMPLETE") for s, v in ok.items()})
    sys.exit(0 if all(ok.values()) else 1)
