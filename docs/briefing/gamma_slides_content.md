# SafeCharge-RL: Reinforcement Learning for EV Smart Charging

Mid-review, 9-10 Oct 2026
Team 32: Adithyan R, Adithya Rajesh, Balu M Krishna
Guide: Dr. Geetha M, Department of CSE (AI), Amrita Vishwa Vidyapeetham
RL problem area: constrained reinforcement learning for sequential decision-making

---

# The Problem: Who Gets Power, and When?

- Many EVs share one transformer, so total power must stay under a grid limit
- Three goals pull apart: low electricity cost, charged before leaving, low battery wear
- The scheduler only sees the departure time the driver reported
- Reports are often wrong (about 3 hours off on average) and could be faked to get served first

**Why RL:** the future is unknown and decisions are made step by step.

**Problem statement:** learn a charging policy that never exceeds the grid cap, serves every car, keeps cost low, and holds up when reported departure times are wrong.

---

# Existing Work and Our Gap

| Work | What it does | Limitation |
|---|---|---|
| Li, Wan, He (2020) | Safe RL for EV charging | One charger, no shared grid cap |
| Chen, Shi (2022) | Lagrangian + SAC | One EV, synthetic data |
| Ting et al. (2024) | Station limit, early departures | Simulated EVs, noise only |
| Li, Sun (2023) | Real Caltech data; user-entered times are unreliable | Tiny station, not about misreporting |
| Ferragut et al. (2021) | Reporting early gets more service | Rule-based policies only |

**Our difference:** a learned policy that drivers can game, tested on real sessions with a shared hard grid cap, against rules and a hindsight-optimal schedule.

---

# MDP Design

| Part | Definition |
|---|---|
| Agent | Station controller |
| Environment | Simulator replaying real charging days |
| State | Per port: energy left, time to reported departure, laxity. Global: price, time of day, last power |
| Action | Fraction (0 to 1) of max power per port, then a safety cut to the grid cap |
| Reward | Minus electricity cost |
| Costs | Unmet energy at departure (budget 2%), battery-wear proxy |
| Episode | One weekday = 96 steps of 15 min. True departure is never shown |

---

# Algorithm and Data

**PPO-Lagrangian:** PPO plus one multiplier per cost. A multiplier rises when its budget is broken.

**Deep Sets network:** one shared encoder per port, so it works for any number of cars.

**Data:** real ACN-Data sessions
- Caltech 31,424 and JPL 33,638 sessions
- Train Jan-Aug 2019, test Nov 2019-Feb 2020 (85 days)

**Our assumptions:** summer time-of-use tariff, grid cap = 20% of typical peak

---

# System Design

(Insert the architecture diagram: Real Sessions, Report Model, Environment, RL Agent, Safety Projection, back to Environment.)

- Report model makes the reported time truthful or misreported
- Safety projection enforces the grid cap as a hard guarantee

---

# Novel Components

1. Reported departure treated as something a driver can manipulate, tested on a learned policy with real data
2. Paired-replay measure of misreporting gain
3. Hard safety projection plus cost budgets for a shared multi-car station
4. Comparison with a hindsight-optimal schedule on real sessions
5. Test of the "serve only until declared departure" fix on a learned policy
6. Extra finding: real driver reports are common and 2-3 hours wrong

---

# Results So Far

(Insert chart: misreporting gain by method, normal and tight grid cap.)

- **Truthful reports:** the learned policy ties EDF on cost; it charges gently (high-power ratio 0.17 vs 0.77)
- **Wrong reports:** unmet energy grows 2.6x for the learned policy, 1.2-1.6x for rules, about 30x for MPC
- **Cheating works:** reporting early gains extra energy in 93-97% of cases for the learned policy
- **Fix:** stopping service at the declared departure removes the gain
- **Transfer:** fine after COVID, fails on JPL

5 seeds, 85 test days, 95% confidence intervals, zero grid violations.

---

# Demo So Far and Next Plan

**Demo:** a busy real day, all methods, no grid violation. Then a cheat test on a crowded day.

**Done**
- Simulator, safety layer, baselines, offline optimum, MPC
- PPO-Lagrangian with 5 seeds
- Experiments E1, E2, E3, E6 and first part of E4
- 108 tests passing, code on GitHub

**Next**
- Finish E4 training variants
- E5 projection ablation and unprojected baseline
- Optional SAC-Lagrangian
- Final report
