"""Receding-horizon MPC baseline (decision-time LP using DECLARED deadlines).

At every step it solves, for the vehicles currently plugged in, an LP over the remaining horizon:
  min  sum_t c(t) sum_i x_it * DELTA + M * sum_i m_i
  s.t. 0 <= x_it <= P_port,  sum_i x_it <= P_grid,  sum_t x_it * DELTA + m_i = r_i(now),  charging only before the declared departure
(the declared departure is the reported one, never the true one; future arrivals are unknown, so none are planned). It applies the first
step of the plan. The price is known (time-of-use). Same LP machinery and penalty M as the offline optimum.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .data import DELTA, T


def mpc_action(env, M=5.0):
    w = env.view()
    occ = np.flatnonzero(w["occ"])
    if len(occ) == 0:
        return np.zeros(env.N)
    t0 = env.t
    price = env.day.price
    rep_end = np.clip(np.ceil(w["rep"][occ]).astype(int), t0 + 1, T)   # last allowed step is rep_end - 1; at least the current step
    cols = [(j, t) for j, k in enumerate(occ) for t in range(t0, rep_end[j])]
    j_of, t_of = np.array([c[0] for c in cols]), np.array([c[1] for c in cols])
    nx, n = len(cols), len(occ)
    c = np.concatenate([price[t_of] * DELTA, np.full(n, M)])
    A_ub = coo_matrix((np.ones(nx), (t_of - t0, np.arange(nx))), shape=(T - t0, nx + n)).tocsr()
    A_eq = coo_matrix((np.concatenate([np.full(nx, DELTA), np.ones(n)]),
                       (np.concatenate([j_of, np.arange(n)]), np.concatenate([np.arange(nx), nx + np.arange(n)]))), shape=(n, nx + n)).tocsr()
    res = linprog(c, A_ub=A_ub, b_ub=np.full(T - t0, env.P_grid), A_eq=A_eq, b_eq=w["r"][occ],
                  bounds=[(0, env.P_port)] * nx + [(0, None)] * n, method="highs")
    u = np.zeros(env.N)
    if not res.success:
        return u
    now = t_of == t0
    p = np.zeros(n)
    p[j_of[now]] = res.x[:nx][now]
    h = w["h"][occ]
    u[occ] = np.divide(p, h, out=np.zeros(n), where=h > 0)
    return np.clip(u, 0, 1)
