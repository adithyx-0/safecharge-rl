import datetime as dt

import numpy as np
import pytest

from safecharge.data import Day, P_PORT, price_curve


def make_day(seed, n=30, date=dt.date(2018, 7, 2)):
    rng = np.random.default_rng(seed)
    arr = np.sort(rng.integers(0, 70, n))
    dur = rng.integers(2, 40, n)
    dep = np.minimum(arr + dur, 96)
    keep = dep > arr
    arr, dep = arr[keep], dep[keep]
    cap = P_PORT * (dep - arr) * 0.25
    e = np.minimum(rng.uniform(2, 30, len(arr)), cap)
    return Day(date, arr, dep, e, price_curve(date, "summer"), list(range(len(arr))))


@pytest.fixture(params=range(8))
def day(request):
    return make_day(request.param)
