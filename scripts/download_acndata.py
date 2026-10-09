"""Download all ACN-Data sessions for Caltech and JPL to data/raw/.

- Token is read from the ACN_TOKEN environment variable (never hard-code it).
- Uses Bearer auth. (acnportal's DataClient.get_sessions uses HTTP Basic auth,
  which returned 401 for our token, so we call the REST API directly.)
- Retries transient 5xx / network errors with exponential backoff.
- Saves a checkpoint on failure; re-running resumes from the failed page.

Usage:
    export ACN_TOKEN=...          # from https://ev.caltech.edu/register
    python scripts/download_acndata.py
"""
import json
import os
import sys
import time

import requests

BASE = os.environ.get("ACN_BASE_URL", "https://ev.caltech.edu/api/v1/")
OUT_DIR = os.environ.get("ACN_OUT_DIR", "data/raw")
SITES = ("caltech", "jpl")
MAX_RETRIES = 6
SKIPPED = []  # (site, offset) of records the server would not return
PAGE_DELAY = 0.4  # seconds between pages, to stay under the rate limit


def get_with_retry(url, headers, retries=MAX_RETRIES):
    """Return a 200 response, or None if we should stop (4xx or retries exhausted)."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"  network error ({e}), retry {attempt + 1}/{retries}")
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            return r
        if 500 <= r.status_code < 600:
            wait = 2 ** attempt
            print(f"  HTTP {r.status_code} (server side), retry {attempt + 1}/{retries} in {wait}s")
            time.sleep(wait)
            continue
        print(f"  HTTP {r.status_code} (client error, not retrying): {r.text[:300]}")
        return None
    print(f"  giving up after {retries} retries")
    return None


def fetch_range_small(site, page, headers, skipped):
    """The size-100 page `page` (1-based) keeps failing server-side. Re-fetch the same records in pieces of 20,
    then one by one. Records that still fail are appended to `skipped` as (site, offset) so none vanish silently."""
    items = []
    first = (page - 1) * 100  # offset of the first record of the failed page
    for k in range(5):
        off = first + 20 * k
        r = get_with_retry(BASE + f"sessions/{site}?max_results=20&page={off // 20 + 1}", headers, retries=2)
        if r is not None:
            items.extend(r.json().get("_items", []))
            continue
        for j in range(20):
            o = off + j
            r1 = get_with_retry(BASE + f"sessions/{site}?max_results=1&page={o + 1}", headers, retries=2)
            if r1 is None:
                skipped.append((site, o))
                print(f"  SKIPPED record at offset {o}")
            else:
                items.extend(r1.json().get("_items", []))
    return items


def download_site(site, headers):
    skipped_all = SKIPPED
    out_path = os.path.join(OUT_DIR, f"{site}_sessions.json")
    progress_path = os.path.join(OUT_DIR, f"{site}_progress.json")

    if os.path.exists(out_path):
        print(f"{site}: {out_path} already exists, skipping (delete it to re-download).")
        return True

    if os.path.exists(progress_path):
        with open(progress_path) as f:
            state = json.load(f)
        sessions, url, page = state["sessions"], state["next_url"], state["page"]
        print(f"{site}: resuming at page {page} with {len(sessions)} sessions already collected.")
    else:
        sessions, url, page = [], BASE + f"sessions/{site}?max_results=100", 0

    skipped_path = os.path.join(OUT_DIR, f"{site}_skipped.json")
    if os.path.exists(skipped_path):
        with open(skipped_path) as f:
            skipped_all.extend(tuple(x) for x in json.load(f))

    def save_progress():
        with open(progress_path, "w") as f:
            json.dump({"sessions": sessions, "next_url": url, "page": page}, f)
        with open(skipped_path, "w") as f:
            json.dump(skipped_all, f)

    while url:
        r = get_with_retry(url, headers)
        if r is None:
            print(f"{site}: page {page + 1} keeps failing at size 100, trying smaller pieces")
            n_skipped_before = len(skipped_all)
            items = fetch_range_small(site, page + 1, headers, skipped_all)
            if not items:  # nothing at all came back: treat as an outage, never as 100 skipped records
                del skipped_all[n_skipped_before:]
                save_progress()
                print(f"{site}: stopped at page {page} with {len(sessions)} sessions. Re-run to resume.")
                return False
            payload = {"_items": items, "_links": {"next": {"href": f"sessions/{site}?max_results=100&page={page + 2}"}}}
            if len(items) + sum(1 for s_ in skipped_all if s_[0] == site and (page * 100) <= s_[1] < (page + 1) * 100) < 100:
                payload["_links"] = {}  # short page = end of data
        else:
            payload = r.json()
        if "_items" not in payload:
            print(f"{site}: no '_items' on page {page}: {str(payload)[:300]}")
            save_progress()
            return False
        sessions.extend(payload["_items"])
        page += 1
        if page % 20 == 0:
            print(f"{site}: {len(sessions)} sessions ({page} pages)")
        next_href = payload.get("_links", {}).get("next", {}).get("href")
        url = BASE + next_href if next_href else None
        time.sleep(PAGE_DELAY)

    with open(skipped_path, "w") as f:
        json.dump(skipped_all, f)
    with open(out_path, "w") as f:
        json.dump(sessions, f)
    if os.path.exists(progress_path):
        os.remove(progress_path)
    print(f"{site}: COMPLETE, {len(sessions)} sessions -> {out_path}")
    return True


def main():
    token = os.environ.get("ACN_TOKEN")
    if not token:
        sys.exit("Set the ACN_TOKEN environment variable first.")
    headers = {"Authorization": f"Bearer {token}"}
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {site: download_site(site, headers) for site in SITES}
    print("\nSummary:")
    for site, ok in results.items():
        print(f"  {site}: {'complete' if ok else 'INCOMPLETE - re-run to resume'}")
    if SKIPPED:
        print(f"WARNING: {len(SKIPPED)} records could not be fetched (offsets): {SKIPPED}")
    print("Expected totals (as of Sep 2026): caltech 31424, jpl 33638. Check the counts above.")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
