import numpy as np
import cvxpy as cp
from safecharge.projection import project

rng = np.random.default_rng(0)


def rand_case():
    n = int(rng.integers(1, 55))
    h = rng.uniform(0, 6.656, n) * (rng.random(n) < 0.8)
    u = rng.random(n)
    p_tilde = u * h
    P_grid = rng.uniform(0.2, 1.0) * max(h.sum(), 1e-3)
    return p_tilde, h, P_grid


def test_feasible_and_bounds():
    for _ in range(500):
        pt, h, G = rand_case()
        p = project(pt, h, G)
        assert np.all(p >= -1e-12) and np.all(p <= h + 1e-12)
        assert p.sum() <= G + 1e-9


def test_identity_when_feasible():
    for _ in range(200):
        pt, h, G = rand_case()
        if pt.sum() <= G:
            assert np.array_equal(project(pt, h, G), pt)


def test_idempotent():
    for _ in range(200):
        pt, h, G = rand_case()
        p = project(pt, h, G)
        assert np.allclose(project(p, h, G), p, atol=1e-9)


def test_matches_qp():
    for _ in range(40):
        pt, h, G = rand_case()
        x = cp.Variable(len(pt))
        cp.Problem(cp.Minimize(cp.sum_squares(x - pt)), [x >= 0, x <= h, cp.sum(x) <= G]).solve(
            solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)  # default tol gives ~1e-5 error
        assert np.allclose(project(pt, h, G), x.value, atol=1e-6)


def test_priority_feasible():
    for _ in range(200):
        pt, h, G = rand_case()
        w = rng.uniform(0.1, 2.0, len(pt))
        p = project(pt, h, G, scheme="priority", weights=w)
        assert np.all(p >= 0) and np.all(p <= h + 1e-12) and p.sum() <= G + 1e-9
