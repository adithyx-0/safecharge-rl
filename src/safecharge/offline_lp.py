"""Offline optimum: full-information LP with true departures.

min sum_t c(t) sum_i x_it * DELTA + M * sum_i m_i
s.t. 0 <= x_it <= P_port for t in [arr_i, dep_i), sum_i x_it <= P_grid, sum_t x_it*DELTA + m_i = e_i, m_i >= 0.
M is in $/kWh and must sit well above the peak price (0.26668). `vehicles` lets the caller drop vehicles the env
could not seat (no free port), so the bound compares like with like.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .data import DELTA, T


def solve_offline(day, P_grid, P_port, M, vehicles=None):
    idx = np.arange(len(day.arr)) if vehicles is None else np.asarray(vehicles)
    n = len(idx)
    cols, veh_of, t_of = 0, [], []
    for j, i in enumerate(idx):
        for t in range(day.arr[i], day.dep[i]):
            veh_of.append(j)
            t_of.append(t)
    nx = len(veh_of)
    veh_of, t_of = np.array(veh_of, int), np.array(t_of, int)
    c = np.concatenate([day.price[t_of] * DELTA, np.full(n, M)])
    # grid rows (inequality)
    A_ub = coo_matrix((np.ones(nx), (t_of, np.arange(nx))), shape=(T, nx + n)).tocsr()
    b_ub = np.full(T, P_grid)
    # energy rows (equality): sum_t x*DELTA + m_j = e_j
    rows = np.concatenate([veh_of, np.arange(n)])
    colsx = np.concatenate([np.arange(nx), nx + np.arange(n)])
    vals = np.concatenate([np.full(nx, DELTA), np.ones(n)])
    A_eq = coo_matrix((vals, (rows, colsx)), shape=(n, nx + n)).tocsr()
    b_eq = day.energy[idx]
    bounds = [(0, P_port)] * nx + [(0, None)] * n
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"offline LP failed: {res.message}")
    x, m = res.x[:nx], res.x[nx:]
    cost = float(np.sum(day.price[t_of] * x * DELTA))
    return dict(cost=cost, unmet=float(m.sum()), objective=float(res.fun), m=m, x=x, veh=veh_of, t=t_of, idx=idx)
