"""PPO-Lagrangian with two constraints (unmet energy, wear proxy). Single file, CleanRL style.

Per update: K episodes (one day each, 96 steps) are rolled out in lockstep; GAE is computed separately for reward and both costs;
the policy maximises  (A_R - sum_k lam_k A_Ck) / (1 + sum_k lam_k)  with each advantage normalised by its batch std (OmniSafe-style);
after the update the multipliers get a dual-ascent step  lam_k <- max(0, lam_k + alpha * (J_k / d_k - 1)),  where J_k is the
(smoothed) mean UNDISCOUNTED episode cost of the batch (relative violation, so alpha does not depend on cost units). Critic targets are
scaled by running per-head scales. Continuous actions a ~ N(mean, std) per occupied port, applied as u = sigmoid(a).
"""
import numpy as np
import torch

from .deepsets import Actor, Critic
from ..baselines import rule_action
from ..data import T
from ..mpc import mpc_action
from ..env import ChargingEnv
from ..metrics import rollout
from ..report_model import reports


def gae(rew, val, gamma, lam):
    """rew, val: (T, K); val has T+1 rows (bootstrap 0 at episode end). Returns advantages (T, K) and returns."""
    adv = np.zeros_like(rew)
    last = 0.0
    for t in reversed(range(rew.shape[0])):
        delta = rew[t] + gamma * val[t + 1] - val[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + val[:-1]


def dual_step(lam, J, d, alpha, clip=(-1.0, 1.0), lam_max=np.inf):
    """lam_k <- clip_to[0, lam_max](lam_k + alpha * clip(J_k / d_k - 1)). The relative violation is clipped so that a huge early
    violation does not wind lam up (it took about 1800 updates to unwind lam = 89 in the first run); lam_max caps the weight on the costs."""
    v = np.clip(np.asarray(J) / np.asarray(d) - 1.0, clip[0], clip[1])
    return np.clip(lam + alpha * v, 0.0, lam_max)


class PPOLag:
    def __init__(self, cfg, n_ports, P_grid, e_ref, reward_scale, d, seed=0):
        self.cfg, self.N = cfg, n_ports
        self.env_kw = dict(N=n_ports, P_grid=P_grid, e_ref=e_ref, reward_scale=reward_scale, scheme=cfg.get("scheme", "uniform"), curtail=cfg.get("curtail", False))
        self.d = np.asarray(d, float)                       # (d1, d2) in episode-cost units
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.actor = Actor(n_ports, cfg["hid"], cfg["use_count"], pool_kind=cfg.get("pool", "mean"))
        self.critic = Critic(n_ports, cfg["hid"], cfg["use_count"], pool_kind=cfg.get("pool", "mean"))
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg["lr"])
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg["lr"])
        self.lam = np.zeros(2)
        self.J_ema = None
        self.scale = np.ones(3)                              # running per-head return scales
        self.envs = [ChargingEnv(**self.env_kw) for _ in range(cfg["n_envs"])]
        self.n_updates = 0
        self.seed = seed
        self.pool = None                                     # empirical report-error pool, only needed for mode M1r

    # ---- rollout ----------------------------------------------------------------------------------------------
    def _episode_reports(self, day, day_idx):
        mix = self.cfg["train_reports"]
        k = self.rng.choice(len(mix), p=np.array([m.get("weight", 1.0) for m in mix]) / sum(m.get("weight", 1.0) for m in mix))
        m = {kk: v for kk, v in mix[k].items() if kk != "weight"}
        return reports(day, m.pop("mode"), int(self.rng.integers(1 << 30)), day_idx, pool=self.pool, **m)

    def collect(self, train_days):
        K, N = self.cfg["n_envs"], self.N
        idx = self.rng.choice(len(train_days), K, replace=len(train_days) < K)
        obs = np.stack([env.reset(options=dict(day=train_days[i], rep=self._episode_reports(train_days[i], int(i))))[0]
                        for env, i in zip(self.envs, idx)])
        O, A, LP = np.zeros((T, K, obs.shape[1]), np.float32), np.zeros((T, K, N), np.float32), np.zeros((T, K), np.float32)
        R, C1, C2 = (np.zeros((T, K)) for _ in range(3))
        for t in range(T):
            o = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                a, lp = self.actor.act(o)
            u = torch.sigmoid(a).numpy().astype(np.float64)
            O[t], A[t], LP[t] = o.numpy(), a.numpy(), lp.numpy()
            nxt = []
            for k, env in enumerate(self.envs):
                ob, r, done, _, info = env.step(u[k])
                R[t, k], C1[t, k], C2[t, k] = r, info["cost1"], info["cost2"]
                nxt.append(ob)
            obs = np.stack(nxt)
        return dict(O=O, A=A, LP=LP, R=R, C=np.stack([C1, C2]))

    def set_lr(self, frac_done):
        """Linear decay from lr to lr * lr_final over training (cfg lr_final, default 1 = constant)."""
        lr = self.cfg["lr"] * (1.0 - (1.0 - self.cfg.get("lr_final", 1.0)) * frac_done)
        for o in (self.opt_a, self.opt_c):
            for g_ in o.param_groups:
                g_["lr"] = lr

    # ---- update -----------------------------------------------------------------------------------------------
    def update(self, b):
        cfg, g, lm = self.cfg, self.cfg["gamma"], self.cfg["gae_lambda"]
        T_, K = b["R"].shape
        O = torch.as_tensor(b["O"].reshape(T_ * K, -1))
        with torch.no_grad():
            v = (self.critic(O).numpy() * self.scale).reshape(T_, K, 3)
        sig = [b["R"], b["C"][0], b["C"][1]]
        advs, rets = [], []
        for h in range(3):
            val = np.concatenate([v[:, :, h], np.zeros((1, K))], 0)
            a_, r_ = gae(sig[h], val, g, lm)
            advs.append(a_), rets.append(r_)
        # critic scales: EMA of the batch return std (floor), then targets in normalised units
        for h in range(3):
            self.scale[h] = 0.9 * self.scale[h] + 0.1 * max(float(rets[h].std()), 1e-2) if self.n_updates else max(float(rets[h].std()), 1e-2)
        nadv = [(a_ - a_.mean()) / (a_.std() + 1e-8) for a_ in advs]
        lam = self.lam
        comb = (nadv[0] - lam[0] * nadv[1] - lam[1] * nadv[2]) / (1.0 + lam.sum())
        tens = lambda x: torch.as_tensor(np.asarray(x).reshape(T_ * K, *np.asarray(x).shape[2:]), dtype=torch.float32)
        A_, LP_, ADV = tens(b["A"]), tens(b["LP"]), tens(comb)
        RET = torch.as_tensor(np.stack([r_.reshape(-1) for r_ in rets], 1) / self.scale, dtype=torch.float32)
        n, mb = T_ * K, cfg["minibatch"]
        stats = dict(pl=0.0, vl=0.0, kl=0.0, ent=0.0, cnt=0)
        for _ in range(cfg["epochs"]):
            perm = torch.randperm(n)
            stop = False
            for s in range(0, n, mb):
                i = perm[s:s + mb]
                mean, log_std, occ = self.actor(O[i])
                lp = Actor.log_prob(A_[i], mean, log_std, occ)
                ratio = (lp - LP_[i]).exp()
                pl = -torch.min(ratio * ADV[i], ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]) * ADV[i]).mean()
                ent = Actor.entropy(log_std, occ).mean()
                if self.n_updates >= cfg.get("critic_warmup", 0):
                    self.opt_a.zero_grad()
                    (pl - cfg["ent_coef"] * ent).backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.opt_a.step()
                vl = ((self.critic(O[i]) - RET[i]) ** 2).mean()
                self.opt_c.zero_grad()
                vl.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.opt_c.step()
                kl = float((LP_[i] - lp.detach()).mean())
                for key, val in (("pl", float(pl)), ("vl", float(vl)), ("kl", kl), ("ent", float(ent))):
                    stats[key] += val
                stats["cnt"] += 1
                if kl > 1.5 * cfg["target_kl"]:
                    stop = True
                    break
            if stop:
                break
        # dual ascent on smoothed undiscounted episode costs
        J = b["C"].sum(1).mean(-1)                           # (2,) undiscounted episode cost (sum over time), mean over the K episodes
        self.J_ema = J if self.J_ema is None else cfg["j_ema"] * self.J_ema + (1 - cfg["j_ema"]) * J
        if self.n_updates >= cfg.get("critic_warmup", 0):
            self.lam = dual_step(self.lam, self.J_ema, self.d, cfg["dual_lr"], tuple(cfg.get("dual_clip", (-1.0, 1.0))), cfg.get("lam_max", np.inf))
        self.n_updates += 1
        c = max(stats.pop("cnt"), 1)
        out = {k: v / c for k, v in stats.items()}
        out.update(J1=float(J[0]), J2=float(J[1]), lam1=float(self.lam[0]), lam2=float(self.lam[1]),
                   ep_return=float(b["R"].sum(0).mean()))
        return out

    # ---- optional warm start: behaviour cloning of a rule baseline (NOT part of the submitted design; reported as such) ----------
    def bc_pretrain(self, train_days, expert="LLF", epochs=30, lr=1e-3, batch=512):
        """Regress the actor mean onto logit(u_expert) on the expert's own trajectories (train days, configured report mixture)."""
        env, O, U = ChargingEnv(**self.env_kw), [], []
        for i, d in enumerate(train_days):
            obs, _ = env.reset(options=dict(day=d, rep=self._episode_reports(d, i)))
            for _ in range(T):
                u = mpc_action(env) if expert == "MPC" else rule_action(env, expert)
                O.append(obs), U.append(u)
                obs, _, done, _, _ = env.step(u)
        O, U = torch.as_tensor(np.array(O), dtype=torch.float32), torch.as_tensor(np.array(U), dtype=torch.float32)
        tgt = torch.logit(U.clamp(0.05, 0.95))
        opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        loss = None
        for _ in range(epochs):
            perm = torch.randperm(len(O))
            for s_ in range(0, len(O), batch):
                i = perm[s_:s_ + batch]
                mean, _, occ = self.actor(O[i])
                loss = (((mean - tgt[i]) ** 2) * occ).sum() / occ.sum().clamp(min=1)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return float(loss)

    # ---- evaluation (deterministic mean action) ---------------------------------------------------------------
    def policy_fn(self, deterministic=True):
        def f(env, obs):
            with torch.no_grad():
                a, _ = self.actor.act(torch.as_tensor(obs, dtype=torch.float32), deterministic=deterministic)
            return torch.sigmoid(a).numpy().astype(np.float64)
        return f

    def evaluate(self, days, mode="M0", seed=12345, **kw):
        env = ChargingEnv(**self.env_kw)
        rows = []
        for i, d in enumerate(days):
            m, _ = rollout(env, d, reports(d, mode, seed, i, pool=self.pool, **kw), self.policy_fn())
            rows.append(m)
        return rows

    # ---- checkpointing ----------------------------------------------------------------------------------------
    def state(self):
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(), opt_a=self.opt_a.state_dict(),
                    opt_c=self.opt_c.state_dict(), lam=self.lam, J_ema=self.J_ema, scale=self.scale, n_updates=self.n_updates,
                    rng=self.rng.bit_generator.state)

    def load(self, st):
        self.actor.load_state_dict(st["actor"]), self.critic.load_state_dict(st["critic"])
        self.opt_a.load_state_dict(st["opt_a"]), self.opt_c.load_state_dict(st["opt_c"])
        self.lam, self.J_ema, self.scale, self.n_updates = st["lam"], st["J_ema"], st["scale"], st["n_updates"]
        self.rng.bit_generator.state = st["rng"]
