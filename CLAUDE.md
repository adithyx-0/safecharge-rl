# SafeCharge-RL: implementation brief

Term project (RL), Team 32, Amrita Vishwa Vidyapeetham, guide Dr. Geetha M. Three students, about one month.
Question studied: how does a learned EV-charging policy behave when drivers' reported departure times are wrong or
deliberately misreported? Real ACN-Data sessions, PPO-Lagrangian with a hard safety projection layer, compared against
rule-based baselines and an offline-optimal schedule.

The written design (Sections VIII-XI) is already submitted. This phase is implementation and experiments.

## Source of truth
- `docs/final_sections_VIII-XI_submitted.pdf` is the submitted design (the `.md` next to it is a text copy; the PDF wins on conflict).
- `docs/proposal_sections_I-VII.md` has the background, literature (Ferragut et al., Li & Sun, Ting et al., Lee/Li/Low), the three hypotheses H1-H3, and the stated limits.
- If code and the submitted design disagree, the design wins. If you think the design should change, stop and ask first. Do not silently deviate.

## Rules
1. Ask before changing anything the design defines (formulation, constants, experiments, baselines).
2. Never invent results. Every number must come from a script in this repo, with a seed and a config file, and must be reproducible.
3. Keep it simple. The faculty want effort on formulation and novelty, solved with existing RL algorithms, not heavy engineering. No frameworks, plugin systems or premature abstraction.
4. The team will write the report text themselves. Output result tables (CSV), figures, and short plain bullet findings. Do not write report prose meant to be pasted into the submission.
5. Never commit the ACN token or the raw data. Token lives in the `ACN_TOKEN` env var. `data/` and `.env` are gitignored.
6. Plan before coding each phase: state what you will build and how you will test it, then wait for a go-ahead.

## Formulation (summary of Sections VIII-XI; the PDF is authoritative)
- Episode = one weekday, T = 96 steps of 15 min (delta = 0.25 h). Station has N ports, per-port max `P_port`, shared grid cap `P_grid`.
- Vehicle i: arrives `t_arr`, needs `e_i` kWh, driver reports `t_rep`, actually leaves at `t_dep` (always, regardless of report). Unmet energy `m_i = max(0, r_i(t_dep))`.
- Energy: `r_i(t+1) = r_i(t) - p_i(t)*delta`, `r_i(t_arr) = e_i`, `0 <= p_i <= min(P_port, r_i/delta)`.
- State: per port `[occupied, r_i, tau_i, laxity_i]` with `tau_i = max(0, t_rep - t)*delta` and `laxity_i = tau_i - r_i/P_port`; global `[price c(t), sin(2*pi*t/T), cos(2*pi*t/T), P(t-1)/P_grid]`. Total 4N + 4. Empty ports zero-padded and masked. The true departure time is never in the state.
- Action: `u in [0,1]^N`. `h_i = occupied_i * min(P_port, r_i/delta)`, `p~_i = u_i*h_i`.
- Safety projection (inside the env step): `p_i = clip(p~_i - mu, 0, h_i)`, `mu >= 0` the smallest value with `sum p_i <= P_grid`; `mu = 0` if already feasible. Find mu by bisection on `[0, max p~]` (the sum is continuous and non-increasing in mu). This is the exact Euclidean projection onto `{0 <= p <= h, sum p <= P_grid}`.
- Reward: `R_t = -c(t) * sum_i p_i * delta`, normalised to roughly unit scale.
- Cost 1 (unmet energy): `c1_t = (sum of m_i for vehicles departing at step t) / e_ref`.
- Cost 2 (high-power proxy for battery wear): `c2_t = (1/N) * sum_i 1[p_i > 0.8*P_port]`.
- Objective: maximise discounted reward subject to `J_Ck <= d_k`, gamma = 0.99, `d1` = 2% of requested energy, `d2` set from the wear proxy of the rule-based baselines. PPO-Lagrangian, multipliers updated by dual ascent after each policy update. SAC-Lagrangian only if time allows.
- Policy: Deep Sets. Shared per-port encoder, masked mean-pool to a context vector, shared head over (port encoding, context, global features) giving mean and std of `u_i`. Critics (reward + 2 cost) share the same structure with a pooled read-out. Memoryless (observation treated as state). Self-attention across ports is a possible later ablation only.
- Report modes (M1 was dropped, see below): M0 truthful; M2 noisy `t_rep = t_dep + eps`, `eps ~ U(-w,w)`, w in {0.5,1,2} h; M3 late `t_rep = t_dep + delta`, `delta ~ U(0,w)` for a share q of vehicles; M4 strategic, a share rho in {0.1,0.25,0.5} reports `t_dep - s`, s in {0.5,1,2} h. Reports are clipped to stay after arrival. Same random draws are reused across methods.
- Baselines: FCFS, EDF (by reported departure), LLF (by reported laxity), unprojected RL (soft grid-violation cost instead of projection), MPC (receding-horizon LP using declared deadlines), offline optimum (LP with true departures).
- Offline LP: minimise `sum_t c(t) sum_i p_i(t) delta + M * sum_i m_i` s.t. `0 <= p_i <= P_port`, `sum_i p_i <= P_grid`, per-vehicle energy balance, charging only inside `[t_arr, t_dep)`. Choose M well above the peak price (peak is $0.26668/kWh) so serving energy always beats saving money.
- Experiments: E1 policy vs baselines under M0; E2 degradation under M2/M3 (increasing w, q); E3 misreporting gain under M4 by paired replay (H1, H2); E4 curtail-at-declared-departure and randomised-report training (H3); E5 projection scheme ablation (uniform cut vs priority-weighted vs no projection); E6 transfer to JPL and post-shift period.
- Metrics: cost per day and per kWh; unmet-energy share U; deadline-miss rate (`E_i < 0.95 e_i`); grid violations (count and max excess); high-power ratio W; Jain fairness; gap to offline optimum; misreporting gain; decision time per step.
- Paired replay (misreporting gain): pick a focal vehicle, run the same day twice with the same policy and the same other reports, once truthful and once reporting `t_dep - s`; the gain is the difference in delivered energy `E_i` (kWh and % of `e_i`).
- Protocol: identical test days and report draws for every method; at least 5 seeds for final numbers (3 for exploration); mean with 95% bootstrap CI over test days.

## Verified facts about the data and environment
- ACN-Data counts (Sep 2026): Caltech 31,424 sessions, JPL 33,638. Session keys: `_id, userInputs, sessionID, stationID, spaceID, siteID, clusterID, connectionTime, disconnectTime, kWhDelivered, doneChargingTime, timezone, userID`.
- API: `https://ev.caltech.edu/api/v1/sessions/{site}?max_results=100`, pagination via `_links.next.href` appended to the base URL. **Use Bearer auth.** acnportal's `DataClient.get_sessions` uses HTTP Basic, which returned 401 with our token. Server throws transient 502s; `scripts/download_acndata.py` retries with backoff and resumes from a checkpoint. A first attempt stopped at about 5,500 Caltech and 6,400 JPL sessions, so **verify the full counts before trusting `data/raw/`**.
- `userInputs.requestedDeparture` was populated in only 0.2% of a 500-session Caltech sample. Recorded driver reports are unusable, so mode M1 is dropped. Do not reintroduce it. The numbering M0, M2, M3, M4 is deliberate.
- `P_port = 6.656 kW` (32 A x 208 V, every port at both sites). N = 54 (Caltech), N = 52 (JPL). Source: `acnportal.acnsim.network.sites.caltech_acn()` / `jpl_acn()`.
- Real transformer references (used only to calibrate kappa in `P_grid = kappa * P_peak`): Caltech 150 kW default vs 54 x 6.656 = 359 kW if every port ran at once; JPL has a 45 kW and a 150 kW group. The scalar `P_grid` is a stated simplification of this.
- Tariff: SCE TOU-EV-4 (March 2019), file `acnportal/signals/tariffs/tariff_schedules/sce_tou_ev_4_march_2019.json`. Summer weekday (Jun 1 - Sep 30), period starts at hours [0, 8, 12, 18, 23], $/kWh [0.05623, 0.0925, 0.26668, 0.0925, 0.05623]. Winter weekday (Oct 1 - May 31) same boundaries, [0.06087, 0.07492, 0.0869, 0.07492, 0.06087]. Weekends flat.
- The user is on a MacBook Air (Apple Silicon), Python 3.12 in a `.venv`. CPU-only is the plan, see Compute.

## Open decisions: settle with the user before locking
1. **Which days get the summer tariff.** The design says the main experiments use the summer schedule. Either (a) use only Jun-Sep weekdays with the real dates, or (b) apply the summer price curve to all weekdays as a fixed experimental tariff. (a) is the real tariff but uses about a third of the days; (b) uses more data but is a controlled assumption. Measure how many days each leaves, then ask.
2. **Sessions crossing midnight.** Measure the fraction first. Options: drop them, or clip departure to the day end. Document whichever is chosen and report the share affected.
3. **DST transition days** have 92 or 100 steps, not 96. Simplest: drop them.
4. **15-minute grid rounding.** Suggested: arrival rounded up, departure rounded down (conservative), then clip `e_i` to `P_port * window`. The uncontrolled-replay check below must report the resulting aggregate energy shortfall versus recorded `kWhDelivered`.
5. **Local time.** Bucket days and tariff periods in local time using each session's `timezone` field, not UTC.
6. **Simulator.** The design says "build on ACN-Sim, thin Gymnasium wrapper as fallback". ACN-Sim's simulator works in amps and network constraint matrices, and its data client failed auth. Recommended: write a small custom Gymnasium env (pure numpy energy bookkeeping, easy to vectorise), and use acnportal only for the static reference data above. This falls under the design's own fallback clause, but tell the user so it is stated honestly.
7. **Constrained PPO.** The design says "existing implementation". Time-box (about 2 hours) a check of whether a library PPO-Lagrangian (for example OmniSafe) runs with a custom Gymnasium env and the custom Deep Sets policy; compatibility is unverified. If not, adapt a single-file PPO (CleanRL style) and add the Lagrange multipliers, which is simple and easy to debug. Ask which the user prefers.
8. **MPC baseline.** Section XI lists it as a baseline, but it is the most skippable item. It reuses the offline LP code in a receding horizon, so do it right after the LP if cheap; otherwise tell the user so the document can be adjusted.

## Build order (each phase must work on its own before the next)
**Phase 0: data, environment, no RL.**
- Data audit script: counts, sessions per year and month (to choose the chronological split inside one stable pre-2020 period and a post-shift test set), zero-energy sessions, midnight crossings, `e_i > P_port * duration` cases, userInputs coverage. Print it as a table and ask the user to pick split dates.
- Loader and preprocessing; report-mode generators (M0, M2, M3, M4) with seeded draws.
- Environment with the projection layer; rule-based baselines FCFS/EDF/LLF; offline LP; metrics.

**Phase 1:** PPO-Lagrangian trained and tested under M0 (E1). **Phase 2:** the same policy evaluated zero-shot under M2/M3 (E2). **Phase 3:** M4 plus paired replay (E3), the central novelty claim. **Phase 4:** randomised-report training and curtail-at-declared-departure (E4). **Phase 5, as time allows:** E5, E6, SAC-Lagrangian, MPC, self-attention.

## Tests that must pass (these are also the proof the team can show)
- Projection: always `0 <= p <= h` and `sum p <= P_grid`; identity when the request is already feasible; idempotent; matches a QP solver (scipy or cvxpy) on random inputs to 1e-6.
- Energy: no vehicle ever receives more than `e_i`; `m_i` is computed at the true departure; a vehicle never charges outside its window.
- Uncontrolled replay (every vehicle at full rate on arrival, no grid cap) reproduces recorded `kWhDelivered` in aggregate; report the discrepancy (see decision 4).
- Offline LP is a lower bound: on every test day, its objective (cost + M * unmet) is <= that of FCFS, EDF, LLF and any RL policy.
- Paired replay with `s = 0` gives zero gain exactly.
- Same seed and config give identical results.
- Stratify unmet energy by session duration (short stays vs long) to check the policy does not quietly neglect short-stay vehicles.

## Implementation guidance from the design review (recommended, not in the submitted text)
- Normalise state features, and use running normalisers for reward and both costs; otherwise the dual step size is brittle.
- gamma = 0.99 over 96 steps discounts early actions to about 38% by the end of the day, and unmet-energy cost only arrives at departure. Treat gamma as something to ablate (0.99 vs 0.999) rather than fixed folklore.
- Projection scheme (E5): keep the uniform cut as the main scheme. Implement the priority-weighted variant behind the same interface so E5 is a config switch.
- Report per-experiment tables with the same column set so results are directly comparable.

## Compute
- Workloads are tiny (small MLPs, 96-step episodes, N <= 54, numpy env). CPU is the right device; a GPU will not help. PyTorch CPU build, `torch.set_num_threads` set explicitly.
- Time the environment (steps/s) and one PPO update before planning the run budget. Vectorise the env or run seeds in parallel processes, but cap workers, because the laptop is fanless and will throttle under sustained load.
- Make training resumable (checkpoint every N updates, log to CSV). Use `caffeinate -i python ...` for long runs so macOS does not sleep.
- Colab or Kaggle only as overflow for extra seeds, not as the main environment.

## Suggested layout (create as needed)
```
safecharge_rl/
  CLAUDE.md  requirements.txt  .env.example  .gitignore
  docs/        submitted design, proposal, diagram
  data/raw/    caltech_sessions.json, jpl_sessions.json   (gitignored)
  scripts/     download_acndata.py (done and tested), audit_data.py, run_experiment.py, make_tables.py
  src/safecharge/  data.py, report_model.py, env.py, projection.py, baselines.py, offline_lp.py, metrics.py, agents/
  configs/     one yaml per experiment
  tests/       test_projection.py, test_env.py, test_lp_bound.py, ...
  results/     CSV tables and figures (small ones only in git)
```
