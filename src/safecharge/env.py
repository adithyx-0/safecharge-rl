"""Single-station charging env (Gymnasium API, numpy bookkeeping). One episode = one day, T = 96 steps of 15 min.

Timing: a vehicle with window [arr, dep) is plugged in at the start of step `arr`, can charge in steps arr..dep-1,
and unplugs after step dep-1 (true departure; the policy only ever sees the reported one). Unmet energy m_i is
measured then. Observation per CLAUDE.md: per port [occupied, r, tau, laxity], then global [price, sin, cos, P(t-1)/P_grid].
Features are divided by fixed constants (OBS_*) so they are roughly unit scale (PROVISIONAL scaling).
`info` carries the unscaled physical quantities. Reward is returned in $ / reward_scale.
"""
import gymnasium as gym
import numpy as np

from .data import DELTA, P_PORT, T
from .projection import project

OBS_R, OBS_TAU, OBS_PRICE = 30.0, 24.0, 0.27  # kWh, hours, $/kWh


class ChargingEnv(gym.Env):
    def __init__(self, N, P_grid, P_port=P_PORT, e_ref=1.0, reward_scale=1.0, scheme="uniform", curtail=False):
        self.N, self.P_grid, self.P_port = N, P_grid, P_port
        self.curtail = curtail  # E4 mitigation: a vehicle receives no power in steps t >= ceil(reported departure)
        self.e_ref, self.reward_scale, self.scheme = e_ref, reward_scale, scheme
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (4 * N + 4,), np.float64)
        self.action_space = gym.spaces.Box(0.0, 1.0, (N,), np.float64)

    # ---- episode setup -------------------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.day, self.rep = options["day"], np.asarray(options["rep"], dtype=float)
        n = len(self.day.arr)
        self.t = 0
        self.vid = -np.ones(self.N, int)                 # vehicle index per port, -1 = empty
        self.r = np.zeros(self.N)
        self.E = np.zeros(n)                             # delivered energy per vehicle
        self.m = np.zeros(n)                             # unmet energy at true departure
        self.dropped = np.zeros(n, bool)                 # no free port on arrival
        self.next_arr = 0
        self.last_power = 0.0
        self.power_log = np.zeros(T)
        self.port_power_log = []                         # (t, vehicle, kW) rows with p>0, for metrics
        self._plug_in()
        return self._obs(), {}

    def _plug_in(self):
        d = self.day
        while self.next_arr < len(d.arr) and d.arr[self.next_arr] <= self.t:
            i = self.next_arr
            free = np.flatnonzero(self.vid < 0)
            if len(free) == 0:
                self.dropped[i] = True
                self.m[i] = d.energy[i]
            else:
                self.vid[free[0]] = i
                self.r[free[0]] = d.energy[i]
            self.next_arr += 1

    # ---- views ---------------------------------------------------------------------------------------------
    def view(self):
        """What a policy / rule baseline may use: reported (never true) departure, per port."""
        occ = self.vid >= 0
        v = np.where(occ, self.vid, 0)
        tau = np.where(occ, np.maximum(0.0, self.rep[v] - self.t) * DELTA, 0.0)
        r = np.where(occ, self.r, 0.0)
        lax = np.where(occ, tau - r / self.P_port, 0.0)
        h = occ * np.minimum(self.P_port, r / DELTA)
        if self.curtail:
            h = h * (self.t < np.ceil(np.where(occ, self.rep[v], 0.0)))
        arr = np.where(occ, self.day.arr[v], 0)
        return dict(occ=occ, r=r, tau=tau, laxity=lax, h=h, arr=arr, rep=np.where(occ, self.rep[v], 0.0))

    def _obs(self):
        w = self.view()
        ports = np.stack([w["occ"].astype(float), w["r"] / OBS_R, w["tau"] / OBS_TAU, w["laxity"] / OBS_TAU], 1)
        price = self.day.price[min(self.t, T - 1)]
        ang = 2 * np.pi * self.t / T
        g = [price / OBS_PRICE, np.sin(ang), np.cos(ang), self.last_power / self.P_grid]
        return np.concatenate([ports.ravel(), g])

    # ---- dynamics ------------------------------------------------------------------------------------------
    def step(self, u):
        w = self.view()
        h = w["h"]
        p_tilde = np.clip(np.asarray(u, dtype=float), 0.0, 1.0) * h
        weights = 1.0 + np.maximum(w["laxity"], 0.0)  # used only by scheme='priority' (PROVISIONAL weighting)
        p = project(p_tilde, h, self.P_grid, self.scheme, weights)

        occ = self.vid >= 0
        dE = p * DELTA
        self.r = np.maximum(self.r - dE, 0.0)
        self.r[self.r < 1e-9] = 0.0
        np.add.at(self.E, self.vid[occ], dE[occ])
        total = float(p.sum())
        self.power_log[self.t] = total
        for k in np.flatnonzero(p > 0):
            self.port_power_log.append((self.t, self.vid[k], p[k]))
        price = self.day.price[self.t]
        dollars = price * total * DELTA

        # departures at the end of step t
        c1 = 0.0
        for k in np.flatnonzero(occ):
            i = self.vid[k]
            if self.day.dep[i] == self.t + 1:
                self.m[i] = self.r[k]
                c1 += self.m[i]
                self.vid[k], self.r[k] = -1, 0.0
        c1 /= self.e_ref
        c2 = float(np.sum(p > 0.8 * self.P_port)) / self.N

        self.last_power = total
        self.t += 1
        done = self.t >= T
        if not done:
            self._plug_in()
        info = dict(cost1=c1, cost2=c2, dollars=dollars, power=total, p=p, p_tilde=p_tilde, price=price)
        return self._obs(), -dollars / self.reward_scale, done, False, info
