import numpy as np

from safecharge.agents.ppo_lag import dual_step, gae


def test_gae_matches_hand_computation():
    rew = np.array([[1.0], [2.0], [3.0]])
    val = np.array([[0.5], [1.0], [1.5], [0.0]])
    g, lm = 0.9, 0.8
    d = rew[:, 0] + g * val[1:, 0] - val[:-1, 0]
    a2 = d[2]
    a1 = d[1] + g * lm * a2
    a0 = d[0] + g * lm * a1
    adv, ret = gae(rew, val, g, lm)
    assert np.allclose(adv[:, 0], [a0, a1, a2]) and np.allclose(ret[:, 0], adv[:, 0] + val[:-1, 0])


def test_gae_lambda1_gamma1_is_reward_to_go_minus_value():
    rew = np.random.default_rng(0).random((10, 3))
    val = np.random.default_rng(1).random((11, 3))
    val[-1] = 0
    adv, ret = gae(rew, val, 1.0, 1.0)
    assert np.allclose(ret, np.cumsum(rew[::-1], 0)[::-1])


def test_dual_ascent_rises_when_violated_falls_when_satisfied_never_negative():
    lam = np.zeros(2)
    d = np.array([1.0, 2.0])
    for _ in range(10):
        lam = dual_step(lam, [2.0, 4.0], d, 0.1)          # both 100% over budget
    assert np.allclose(lam, 1.0)
    for _ in range(100):
        lam = dual_step(lam, [0.0, 0.0], d, 0.1)          # far under budget
    assert np.all(lam == 0.0)
    assert np.allclose(dual_step(np.array([0.3, 0.3]), d, d, 0.5), 0.3)  # exactly at budget: unchanged


def test_dual_step_clips_violation_and_caps_lambda():
    d = np.array([1.0, 1.0])
    assert np.allclose(dual_step(np.zeros(2), [50.0, 50.0], d, 0.1), 0.1)          # huge violation counts as 1.0 only
    lam = np.zeros(2)
    for _ in range(100):
        lam = dual_step(lam, [5.0, 5.0], d, 0.5, lam_max=3.0)
    assert np.allclose(lam, 3.0)
