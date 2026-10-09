import torch

from safecharge.agents.deepsets import Actor, Critic

N = 12


def rand_obs(B, n_occ, seed=0):
    g = torch.Generator().manual_seed(seed)
    ports = torch.zeros(B, N, 4)
    for b in range(B):
        idx = torch.randperm(N, generator=g)[:n_occ]
        ports[b, idx, 0] = 1.0
        ports[b, idx, 1:] = torch.randn(n_occ, 3, generator=g)
    glob = torch.randn(B, 4, generator=g)
    return torch.cat([ports.reshape(B, -1), glob], -1)


def permute(obs, perm):
    ports = obs[:, : 4 * N].reshape(-1, N, 4)[:, perm]
    return torch.cat([ports.reshape(obs.shape[0], -1), obs[:, 4 * N:]], -1)


import pytest


@pytest.mark.parametrize("kind", ["mean", "meanmax", "attn"])
def test_permutation_equivariance_actor_and_invariance_critic(kind):
    torch.manual_seed(0)
    actor, critic = Actor(N, pool_kind=kind), Critic(N, pool_kind=kind)
    obs = rand_obs(5, 6)
    perm = torch.randperm(N)
    m1, s1, _ = actor(obs)
    m2, s2, _ = actor(permute(obs, perm))
    assert torch.allclose(m1[:, perm], m2, atol=1e-5) and torch.allclose(s1[:, perm], s2, atol=1e-5)
    assert torch.allclose(critic(obs), critic(permute(obs, perm)), atol=1e-5)


@pytest.mark.parametrize("kind", ["mean", "meanmax", "attn"])
def test_empty_ports_do_not_change_outputs_or_logprob(kind):
    torch.manual_seed(0)
    actor, critic = Actor(N, pool_kind=kind), Critic(N, pool_kind=kind)
    obs = rand_obs(4, 5)
    obs2 = obs.clone()
    ports = obs2[:, : 4 * N].reshape(-1, N, 4)
    empty = ports[..., 0] == 0
    ports[..., 1:][empty] = 99.0                      # garbage in empty ports; the mask must remove it
    obs2[:, : 4 * N] = ports.reshape(4, -1)
    m1, s1, occ = actor(obs)
    m2, s2, _ = actor(obs2)
    a = torch.randn(4, N)
    assert torch.allclose(Actor.log_prob(a, m1, s1, occ), Actor.log_prob(a, m2, s2, occ), atol=1e-5)
    assert torch.allclose(critic(obs), critic(obs2), atol=1e-5)


def test_logprob_matches_torch_normal_on_occupied_only():
    torch.manual_seed(0)
    actor = Actor(N)
    obs = rand_obs(3, 4)
    mean, log_std, occ = actor(obs)
    a = torch.randn(3, N)
    ref = (torch.distributions.Normal(mean, log_std.exp()).log_prob(a) * occ).sum(-1)
    assert torch.allclose(Actor.log_prob(a, mean, log_std, occ), ref, atol=1e-5)
    ent = (torch.distributions.Normal(mean, log_std.exp()).entropy() * occ).sum(-1)
    assert torch.allclose(Actor.entropy(log_std, occ), ent, atol=1e-5)


def test_works_with_different_number_of_vehicles_and_count_switch():
    torch.manual_seed(0)
    for use_count, kind in ((True, "mean"), (False, "mean"), (True, "meanmax"), (True, "attn")):
        actor, critic = Actor(N, use_count=use_count, pool_kind=kind), Critic(N, use_count=use_count, pool_kind=kind)
        for k in (0, 1, N):
            obs = rand_obs(2, k)
            m, s, occ = actor(obs)
            assert torch.isfinite(m).all() and torch.isfinite(critic(obs)).all()
