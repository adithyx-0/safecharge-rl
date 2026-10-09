import numpy as np
import pytest

from safecharge.baselines import rule_action
from safecharge.data import DELTA, P_PORT, T
from safecharge.env import ChargingEnv
from safecharge.metrics import paired_gain, rollout
from safecharge.offline_lp import solve_offline
from safecharge.report_model import reports

N = 54
M_PEN = 5.0


def peak_full_rate(day):
    occ = np.zeros(T)
    for a, b in zip(day.arr, day.dep):
        occ[a:b] += P_PORT
    return occ.max()


def mkenv(day, kappa=0.5):
    return ChargingEnv(N, kappa * peak_full_rate(day), e_ref=day.energy.mean())


def rand_policy(seed):
    rng = np.random.default_rng(seed)
    return lambda env, obs: rng.random(env.N)


def test_energy_invariants_and_window(day):
    env = mkenv(day)
    rep = reports(day, "M2", 0, 0, w=1.0)
    for kind in ("FCFS", "EDF", "LLF"):
        _, env = rollout(env, day, rep, lambda e, o, k=kind: rule_action(e, k))
        assert np.all(env.E <= day.energy + 1e-9)               # never more than e_i
        assert np.allclose(env.m, np.maximum(0, day.energy - env.E), atol=1e-9)  # m_i at the TRUE departure
        log = np.array(env.port_power_log)
        if len(log):
            t, v = log[:, 0].astype(int), log[:, 1].astype(int)
            assert np.all(t >= day.arr[v]) and np.all(t < day.dep[v])  # only charges inside its window
        assert env.power_log.max() <= env.P_grid + 1e-9


def test_true_departure_not_in_state(day):
    env = mkenv(day)
    rep_a = day.dep.astype(float)
    rep_b = rep_a.copy()
    obs_a, _ = env.reset(options=dict(day=day, rep=rep_a))
    # change only the TRUE departure but keep the report identical: observation must be identical
    import copy
    day2 = copy.deepcopy(day)
    day2.dep = np.minimum(day.dep + 3, 96)
    obs_b, _ = env.reset(options=dict(day=day2, rep=rep_b))
    assert np.array_equal(obs_a, obs_b)


def test_obs_shape():
    day = __import__("conftest").make_day(0)
    env = mkenv(day)
    obs, _ = env.reset(options=dict(day=day, rep=day.dep.astype(float)))
    assert obs.shape == (4 * N + 4,)


def test_uncontrolled_delivers_everything(day):
    env = ChargingEnv(N, 1e9, e_ref=1.0)
    _, env = rollout(env, day, day.dep.astype(float), lambda e, o: np.ones(e.N))
    assert np.allclose(env.E, day.energy, atol=1e-9)


@pytest.mark.parametrize("kappa", [0.3, 0.5, 0.8])
def test_offline_lp_is_lower_bound(day, kappa):
    env = mkenv(day, kappa)
    rep = day.dep.astype(float)
    lp = solve_offline(day, env.P_grid, P_PORT, M_PEN)
    pols = {k: (lambda e, o, k=k: rule_action(e, k)) for k in ("FCFS", "EDF", "LLF")}
    pols["rand0"], pols["rand1"] = rand_policy(0), rand_policy(1)
    for name, pol in pols.items():
        m, env = rollout(env, day, rep, pol)
        obj = m["cost"] + M_PEN * np.maximum(day.energy - env.E, 0).sum()
        assert lp["objective"] <= obj + 1e-6, (name, lp["objective"], obj)


def test_paired_replay_s0_is_exactly_zero(day):
    env = mkenv(day)
    base = reports(day, "M4", 0, 0, rho=0.25, s=1.0)
    for focal in range(min(5, len(day.arr))):
        for kind in ("EDF", "LLF"):
            g, p = paired_gain(env, day, base, lambda e, o, k=kind: rule_action(e, k), focal, 0.0)
            assert g == 0.0 and p == 0.0


def test_determinism(day):
    def run():
        env = mkenv(day)
        m, _ = rollout(env, day, reports(day, "M2", 7, 3, w=1.0), lambda e, o: rule_action(e, "LLF"))
        m.pop("ms_per_step")
        return m
    assert run() == run()


def test_curtail_stops_service_at_reported_departure(day):
    rep = reports(day, "M2", 0, 0, w=2.0)
    env = ChargingEnv(N, 1e9, e_ref=1.0, curtail=True)
    _, env = rollout(env, day, rep, lambda e, o: np.ones(e.N))
    log = np.array(env.port_power_log)
    if len(log):
        t, v = log[:, 0].astype(int), log[:, 1].astype(int)
        assert np.all(t < np.ceil(rep[v]))                   # never charged at or after the reported departure
        assert np.all(t < day.dep[v])                        # and still never after the true one
    assert np.all(env.E <= day.energy + 1e-9)


def test_curtail_removes_misreporting_gain_for_edf(day):
    # with curtailment, reporting earlier can only reduce (never increase) the focal vehicle's delivered energy under a deterministic rule
    env = ChargingEnv(N, 0.3 * peak_full_rate(day), e_ref=1.0, curtail=True)
    base = reports(day, "M0", 0, 0)
    for focal in range(min(6, len(day.arr))):
        g, _ = paired_gain(env, day, base, lambda e, o: rule_action(e, "EDF"), focal, 1.0)
        assert g <= 1e-9
