import numpy as np
from safecharge.data import Day, price_curve
import datetime as dt
from safecharge.report_model import reports

rng = np.random.default_rng(1)
n = 40
arr = rng.integers(0, 60, n)
day = Day(dt.date(2018, 7, 2), arr, arr + rng.integers(2, 30, n), rng.uniform(1, 20, n), price_curve(dt.date(2018, 7, 2)), list(range(n)))


def test_m0_truthful():
    assert np.array_equal(reports(day, "M0", 0, 0), day.dep.astype(float))


def test_same_seed_same_draws_and_different_seed_differs():
    a = reports(day, "M2", 3, 5, w=1.0)
    assert np.array_equal(a, reports(day, "M2", 3, 5, w=1.0))
    assert not np.array_equal(a, reports(day, "M2", 4, 5, w=1.0))


def test_clipped_after_arrival():
    for mode, kw in [("M2", dict(w=2.0)), ("M4", dict(rho=0.5, s=2.0))]:
        assert np.all(reports(day, mode, 0, 0, **kw) >= day.arr)


def test_m2_bounds_and_nested():
    r1 = reports(day, "M2", 0, 0, w=0.5)
    r2 = reports(day, "M2", 0, 0, w=2.0)
    ok = day.arr < day.dep - 2.0 / 0.25  # not clipped by arrival even for the largest w
    assert np.all(np.abs(r1 - day.dep)[ok] <= 0.5 / 0.25 + 1e-9)
    assert np.allclose((r2 - day.dep)[ok], 4 * (r1 - day.dep)[ok])  # same draws scaled by w


def test_m3_only_late_and_share():
    r = reports(day, "M3", 0, 0, w=1.0, q=0.0)
    assert np.array_equal(r, day.dep.astype(float))
    r = reports(day, "M3", 0, 0, w=1.0, q=1.0)
    assert np.all(r >= day.dep) and np.all(r <= day.dep + 4 + 1e-9)


def test_m4_selection_nested_in_rho():
    sel_small = reports(day, "M4", 0, 0, rho=0.1, s=1.0) < day.dep
    sel_big = reports(day, "M4", 0, 0, rho=0.5, s=1.0) < day.dep
    ok = day.arr < day.dep - 4
    assert np.all(sel_big[ok] >= sel_small[ok])


def test_focal_zero_shift_is_identity():
    base = reports(day, "M4", 0, 0, rho=0.25, s=1.0)
    # s = 0 override on a truthful vehicle changes nothing
    r0 = reports(day, "M0", 0, 0)
    assert np.array_equal(reports(day, "M0", 0, 0, focal=3, focal_s=0.0), r0)
    assert reports(day, "M0", 0, 0, focal=3, focal_s=1.0)[3] == max(day.dep[3] - 4, day.arr[3])
    del base


def test_m1r_uses_recorded_else_pool_and_is_deterministic():
    import copy
    d = copy.deepcopy(day)
    rec = d.dep.astype(float) + rng.uniform(-6, 6, n)
    rec[::3] = np.nan                                  # a third of the vehicles have no recorded report
    d.rep_rec = rec
    pool = np.array([-4.0, -1.0, 0.0, 2.0, 8.0])
    r = reports(d, "M1r", 0, 0, pool=pool)
    have = ~np.isnan(rec)
    assert np.allclose(r[have], np.maximum(rec[have], d.arr[have]))          # recorded value used (clipped after arrival)
    err = r[~have] - d.dep[~have]
    assert np.all(np.isin(np.round(err[r[~have] > d.arr[~have]], 6), pool))  # imputed error comes from the pool
    assert np.array_equal(r, reports(d, "M1r", 0, 0, pool=pool))
    assert not np.array_equal(r[~have], reports(d, "M1r", 1, 0, pool=pool)[~have])
