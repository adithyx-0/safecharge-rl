"""Deep Sets actor and critic over ports. Observation layout (see env.py): N rows of [occupied, r, tau, laxity] then 4 globals.

Actor: shared per-port encoder -> masked mean-pool (+ occupied count) -> shared head over (port encoding, context, globals)
giving mean and log-std of a_i; the applied action is u_i = sigmoid(a_i) (Gaussian in a, so the policy is a proper density).
Critic: same encoder, masked mean-pool + count + globals -> MLP -> 3 values (reward, unmet-energy cost, wear cost).
Empty ports are masked everywhere: they do not enter the pool and carry no log-prob (their action has no effect).
`use_count=False` removes the occupied count from the pooled context (pure mean-pool, as literally described in the design).
"""
import torch
import torch.nn as nn


def mlp(sizes, last_gain=1.0):
    layers = []
    for i in range(len(sizes) - 1):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        nn.init.orthogonal_(lin.weight, gain=(last_gain if i == len(sizes) - 2 else 2 ** 0.5))
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        if i < len(sizes) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def split_obs(obs, n_ports):
    ports = obs[..., : 4 * n_ports].reshape(*obs.shape[:-1], n_ports, 4)
    return ports, obs[..., 4 * n_ports:], ports[..., 0]


def pool(z, occ, n_ports, use_count, kind="mean"):
    cnt = occ.sum(-1, keepdim=True)
    ctx = (z * occ.unsqueeze(-1)).sum(-2) / cnt.clamp(min=1.0)
    if kind == "meanmax":     # ablation: also the per-feature max over occupied ports (sees the most urgent vehicle)
        ctx = torch.cat([ctx, torch.where(occ.unsqueeze(-1) > 0, z, torch.full_like(z, -1e4)).max(-2).values.clamp(min=-1.0)], -1)
    return torch.cat([ctx, cnt / n_ports], -1) if use_count else ctx


class SelfAttn(nn.Module):
    """One masked self-attention layer across occupied ports (ablation; the design's later extension). Output has the same shape as z."""

    def __init__(self, hid):
        super().__init__()
        self.q, self.k, self.v, self.o = (nn.Linear(hid, hid) for _ in range(4))
        self.scale = hid ** -0.5

    def forward(self, z, occ):
        att = (self.q(z) @ self.k(z).transpose(-1, -2)) * self.scale
        att = att.masked_fill(occ.unsqueeze(-2) == 0, -1e9).softmax(-1)
        return z + torch.tanh(self.o(att @ self.v(z))) * occ.unsqueeze(-1)


class Actor(nn.Module):
    def __init__(self, n_ports, hid=64, use_count=True, init_log_std=-0.5, pool_kind="mean"):
        super().__init__()
        self.n, self.use_count, self.pool_kind = n_ports, use_count, pool_kind
        self.enc = mlp([4, hid, hid])
        self.attn = SelfAttn(hid) if pool_kind == "attn" else None
        ctx_dim = hid * (2 if pool_kind == "meanmax" else 1) + int(use_count)
        self.head = mlp([hid + ctx_dim + 4, hid, 2], last_gain=0.01)
        self.init_log_std = init_log_std

    def forward(self, obs):
        ports, g, occ = split_obs(obs, self.n)
        z = torch.tanh(self.enc(ports)) * occ.unsqueeze(-1)
        if self.attn is not None:
            z = self.attn(z, occ)
        ctx = pool(z, occ, self.n, self.use_count, self.pool_kind)
        h = self.head(torch.cat([z, ctx.unsqueeze(-2).expand(*z.shape[:-1], ctx.shape[-1]),
                                 g.unsqueeze(-2).expand(*z.shape[:-1], g.shape[-1])], -1))
        mean = h[..., 0]
        log_std = (self.init_log_std + h[..., 1]).clamp(-4.0, 1.0)
        return mean, log_std, occ

    def act(self, obs, deterministic=False):
        mean, log_std, occ = self(obs)
        a = mean if deterministic else mean + torch.randn_like(mean) * log_std.exp()
        return a, self.log_prob(a, mean, log_std, occ)

    @staticmethod
    def log_prob(a, mean, log_std, occ):
        lp = -0.5 * ((a - mean) / log_std.exp()) ** 2 - log_std - 0.9189385332046727
        return (lp * occ).sum(-1)

    @staticmethod
    def entropy(log_std, occ):
        return ((log_std + 1.4189385332046727) * occ).sum(-1)


class Critic(nn.Module):
    """Outputs 3 values in normalised units; the algorithm multiplies by its running per-head scale."""

    def __init__(self, n_ports, hid=64, use_count=True, n_heads=3, pool_kind="mean"):
        super().__init__()
        self.n, self.use_count, self.pool_kind = n_ports, use_count, pool_kind
        self.enc = mlp([4, hid, hid])
        self.attn = SelfAttn(hid) if pool_kind == "attn" else None
        ctx_dim = hid * (2 if pool_kind == "meanmax" else 1) + int(use_count)
        self.head = mlp([ctx_dim + 4, 2 * hid, 2 * hid, n_heads], last_gain=1.0)

    def forward(self, obs):
        ports, g, occ = split_obs(obs, self.n)
        z = torch.tanh(self.enc(ports)) * occ.unsqueeze(-1)
        if self.attn is not None:
            z = self.attn(z, occ)
        return self.head(torch.cat([pool(z, occ, self.n, self.use_count, self.pool_kind), g], -1))
