import numpy as np

from safecharge.data import P_PORT
from safecharge.env import ChargingEnv
from safecharge.metrics import rollout
from safecharge.mpc import mpc_action
from safecharge.offline_lp import solve_offline
from safecharge.report_model import reports
from conftest import make_day

M = 5.0


def test_mpc_with_truthful_reports_is_feasible_and_not_better_than_offline():
    for seed in range(4):
        day = make_day(seed)
        occ = np.zeros(96)
        for a, b in zip(day.arr, day.dep):
            occ[a:b] += P_PORT
        Pg = 0.4 * occ.max()
        env = ChargingEnv(54, Pg, e_ref=day.energy.mean())
        m, env = rollout(env, day, day.dep.astype(float), lambda e, o: mpc_action(e, M))
        lp = solve_offline(day, Pg, P_PORT, M)
        obj = m["cost"] + M * np.maximum(day.energy - env.E, 0).sum()
        assert lp["objective"] <= obj + 1e-6                    # offline optimum is a lower bound for MPC too
        assert m["grid_viol"] == 0 and np.all(env.E <= day.energy + 1e-9)


def test_mpc_with_no_future_arrivals_and_one_vehicle_matches_offline():
    day = make_day(0, n=1)
    env = ChargingEnv(54, 1e9, e_ref=day.energy.mean())
    m, env = rollout(env, day, day.dep.astype(float), lambda e, o: mpc_action(e, M))
    lp = solve_offline(day, 1e9, P_PORT, M)
    assert abs(m["cost"] - lp["cost"]) < 1e-6                   # single vehicle: nothing unknown, so MPC = offline
