# SafeCharge-RL: working log

Newest entries at the bottom. Every change, decision, test result and open question goes here.
Provisional = my default so the code can run; NOT a locked decision. The user decides.

## 2026-10-04 session 1 (user away ~30 min, authorised Phase 0 work)

### Setup
- Unpacked `safecharge_rl_project.zip` into `Project/safecharge_rl/` (the working dir from now on). The zip itself and the
  root-level `CLAUDE.md`, `test.ipynb`, `acndata_raw/` are left untouched.
- System python3 is 3.14; created `.venv` with Python 3.12.13 via `uv`. Installed numpy 2.5.3, torch 2.14.1, scipy,
  pandas, gymnasium, matplotlib, pyyaml, pytest, requests, acnportal, cvxpy (unpinned, as in requirements.txt).
- Copied the old partial checkpoint `acndata_raw/caltech_progress.json` (7,800 Caltech sessions, page 78) into
  `data/raw/` so the download resumes from there instead of restarting.
- Started `scripts/download_acndata.py` in the background (log: `data/download.log`). The token was read from the
  `ACN_TOKEN` env var of that one process only (taken from the user's `test.ipynb`); it was not written to any new file.

### Flag for the user
- **The ACN token is hardcoded in `test.ipynb` (several cells).** CLAUDE.md rule 5 says it must live only in the env
  var. Recommend: delete those cells' token strings and regenerate the token at ev.caltech.edu/register. Not touched by me.


### Code written (all under `src/safecharge/`, tests under `tests/`; run with `.venv/bin/python -m pytest tests -q`)
- `projection.py`: bisection projection. `scheme='uniform'` is the design's exact Euclidean projection; `'priority'` is the
  E5 variant behind the same interface (cut = mu * w_i). Priority weights are PROVISIONAL: w_i = 1 + max(laxity_i, 0), so
  low-laxity vehicles are cut less. E5 not run yet.
- `data.py`: loader + preprocessing. Local-time day bucketing via each session's `timezone`; DST-change days dropped;
  arrival rounded up, departure rounded down, `e_i` clipped to `P_port * window`; weekdays only. Open decisions are
  options, defaults PROVISIONAL: `midnight='drop'` (alt `'clip'`), `tariff='real'` (alt `'summer'` = summer curve on all weekdays).
  Tariff constants copied from the brief and cross-checked against acnportal's `sce_tou_ev_4_march_2019.json` (match).
- `report_model.py`: M0/M2/M3/M4. One base random stream per (seed, day) independent of mode/parameters => same draws
  for every method and nested across w / q / rho. Reports clipped to stay at or after arrival. M1 not reintroduced.
- `env.py`: numpy Gymnasium env, projection inside `step`. True departure is used only to unplug and compute `m_i`.
  Observation scaling constants (30 kWh, 24 h, $0.27) are PROVISIONAL. Reward is returned unscaled in dollars
  (`reward_scale` param, default 1) and `e_ref` is a parameter: both to be fixed at Phase 1 (not design-defined yet).
  Vehicles that find no free port are marked dropped (count reported in metrics; 0 so far).
- `baselines.py`: FCFS / EDF / LLF, greedy fill up to `P_grid`; they see only reported departures.
- `offline_lp.py`: HiGHS LP exactly as in the design. `M` is a parameter; tests use M = $5/kWh (about 19x the peak price),
  PROVISIONAL.
- `metrics.py`: per-day metrics from XI.A, rollout, bootstrap CI, `paired_gain` (paired replay). Short-vs-long
  stratification bins (stay <2h, 2-4h, 4-8h, >8h) are PROVISIONAL.
- `scripts/audit_data.py`, `scripts/check_uncontrolled.py`.

### Tests (77 passing on synthetic days; real-data checks pending the full download)
- Projection: bounds + grid feasibility (500 random cases), identity when feasible, idempotent, matches cvxpy/CLARABEL QP
  to 1e-6. NOTE: with CLARABEL's default tolerance one case differed by 6e-6, i.e. solver error; with tight tolerances
  the max difference is below 1e-9. The test uses the tight tolerances.
- Report model: M0 identity, seed determinism, clipping, nesting in w and rho, M3 only-late, focal override.
- Env: E_i <= e_i; m_i computed at the true departure; no charging outside [arr, dep); grid never exceeded; the
  observation does not change when only the true departure changes; observation size is 4N+4; uncontrolled run delivers
  every clipped request; paired replay with s = 0 gives exactly 0 gain; identical seed => identical results.
- Offline LP lower bound: objective <= FCFS, EDF, LLF and two random policies on 8 synthetic days x kappa in {0.3, 0.5, 0.8}.
- Tests have teeth: deliberately halving the LP grid cap made 13 tests fail; code restored, 77 pass.
- Sanity: EDF on synthetic days with s = 1 h misreport: gain never negative, positive for 2% of vehicles (max 13% of e_i),
  consistent with Ferragut et al. Synthetic only, NOT a result.
- Speed: env about 32,000 steps/s (random policy, N=54, one core); offline LP about 3 ms/day.

### Observations on the first 7,800 Caltech sessions (2018-04-25 .. 2018-08-25; NOT the full data, no conclusions)
- userInputs populated in 0.96% of these sessions (brief says 0.2% of a 500-session sample); both tiny, M1 stays dropped.
- 9.6% of sessions cross local midnight. After the 15-min conversion, 503 of 5,696 kept sessions had `e_i` clipped.
- Uncontrolled replay: simulated energy is 1.14% below recorded kWh (rounding + clipping), midnight='drop'.
  Every vehicle receives exactly its clipped request (max error 4e-15).
- Max concurrency 44 (<54 ports), so no vehicle was dropped for lack of a port.

### Download status and script change
- Both downloads stalled (Caltech at 7,800, JPL at 5,500 sessions). Probed directly: the server returns HTTP 502 after about 32 s
  for the size-100 page `page=79` of Caltech, while smaller page sizes and other pages return 200. So it is a server-side
  problem tied to that slice of records, not rate limiting.
- CHANGED `scripts/download_acndata.py` (original kept as `scripts/download_acndata.py.orig`): when a size-100 page keeps
  failing, re-fetch the same offsets in pages of 20, then single records (offset = (page-1)*size, so ranges line up).
  Any single record the server still refuses is written to `data/raw/<site>_skipped.json` and printed, never dropped silently.
  If a whole page returns nothing it is treated as an outage: the script stops and the checkpoint is kept (so a server
  outage can never turn into skipped records). This is download tooling only, no design item is touched.

### Download status at 14:09 IST (NOT complete, needs the user's attention)
- Caltech pages 79-82 (offsets 7800-8199) fetched fine in memory. Page 83 (offsets 8200-8299) fails at size 100 and
  at size 20, and 33 single records in it have been refused so far (offsets 8200, 8201, 8202, ..., 8236). The server
  looks degraded in this region, not just a few bad records.
- Each refused record costs about 1 minute of retries, so the fallback is very slow (about 500 records in an hour).
  At this rate the full download would take days. The process is still running; it only writes its checkpoint when it
  stops, so progress since 7,800 lives in memory until then.
- `data/raw/` currently holds only the checkpoints (Caltech 7,800 sessions, JPL 5,500). No complete `*_sessions.json` exists,
  so the real-data audit, the uncontrolled-replay check and the split-date choice are all still waiting.
- Decision needed from the user: wait / retry later (server may recover), or use a smaller time-window download
  (query by date range if the API supports it), or proceed with what we have. Skipped offsets are listed in
  `data/raw/<site>_skipped.json` once the script stops. Any use of partial data must be stated honestly in the report.

### 2026-10-04 token removed from notebook (user approved)
- `../test.ipynb`: 5 cells changed. The hardcoded token string (and the `PASTE_YOUR_TOKEN_HERE` placeholder) is replaced by
  `TOKEN = os.environ["ACN_TOKEN"]`, with `import os` added where missing. Notebook is still valid JSON. No backup copy was
  kept, because a backup would still contain the token.
- Searched the project (excluding `.venv`) for the old token string: no file contains it any more (checked `data/`, logs, scripts, docs).
- The user chose not to rotate the key for now (token only ever lived in that local notebook; folder is not a git repo).
  If the notebook or project is ever shared or pushed, rotate it first.
- Anyone running the notebook now needs `export ACN_TOKEN=...` first.

### 2026-10-04 token stored in `.env`, download restarted by me (user asked: "mask the api, don't lose it")
- Token saved in `safecharge_rl/.env` (`ACN_TOKEN=...`), `chmod 600`, and `.env` is in `.gitignore`. The download is started with
  `set -a; source .env; set +a` so the token is never put on a command line or in a log (checked: not in `data/download.log`).
- Server recovered (page 83 returned 200 in 8 s, then 0.8 s). Restarted `scripts/download_acndata.py` under `caffeinate -i`,
  resuming Caltech at page 79 from the 7,800-session checkpoint. Log: `data/download.log`.
- The earlier degraded run's skipped-offset list is not reused: that run was killed before it saved anything, so nothing was skipped.
- Note: the token was also pasted into the chat transcript by the user, which cannot be masked from here. Rotate it if the transcript is shared.

### 2026-10-04 16:30 IST new downloader: by time window (replaces the offset-paging approach)
- The resumed offset-paging run was again skipping records (133 of about 1,700 in offsets 8,900-9,500; about 8%) and would have
  taken days. I stopped it (killed my own process; it had saved nothing, so nothing partial is used).
- Probed the API: it supports `where=connectionTime>="..." and connectionTime<"..."` and returns `_meta.total` per query.
  April 2019 (1,031 sessions) downloaded in 11 pages / 30 s with no failures, so shallow windowed queries avoid the failing deep pages.
- NEW `scripts/download_acndata_windowed.py`: one query per calendar month, checked against that query's `_meta.total`
  (a mismatch or failure splits the window in half, down to one day), each finished window saved under
  `data/raw/windows/<site>/`, resumable, merged and de-duplicated by `_id`. The merged `<site>_sessions.json` is written ONLY if the merged
  count equals the API's site total (Caltech 31,424 confirmed by the API; JPL expected 33,638). Log: `data/download_windowed.log`.
- Old scripts kept: `scripts/download_acndata.py` (+ `.orig`). The old checkpoints in `data/raw/*_progress.json` are not used.

### 2026-10-04 17:00 IST FULL DATA IN, audit run (Phase 0 data step done; decisions below are the user's)
- Download verified: Caltech 31,424 and JPL 33,638 sessions, each equal to the API's `_meta.total`; no window failed or needed splitting;
  0 duplicate sessionIDs; 0 zero-energy sessions; 0 invalid time ranges. Files: `data/raw/{caltech,jpl}_sessions.json` (gitignored).
  Date span: Caltech 2018-04-25 .. 2021-09-13, JPL 2018-09-05 .. 2021-09-13 (local time).
- Tests: still 77 passing. Real-data uncontrolled replay run for both sites, both midnight options (numbers below).

**FINDING 1 (contradicts the brief): `userInputs.requestedDeparture` is NOT rare.**
- Full-data coverage: Caltech 52.2% of sessions, JPL 93.5%. By quarter at Caltech: 2018Q2 1%, 2018Q3 4%, 2018Q4 48%, then 80-97% every quarter
  from 2019Q1 to 2021Q3. The brief's 0.2% came from a 500-session sample, which (the API returns oldest first) is all from April 2018, before
  the app collected it. Entries look like `{minutesAvailable, requestedDeparture, kWhRequested, WhPerMile, milesRequested, ...}`.
- I did NOT reintroduce M1 and changed nothing in the design. This needs the user's decision (design text says M1 was dropped for sparsity).

**FINDING 2 (contradicts the brief): Caltech has one fast station that is not a 6.656 kW port.**
- Station `2-39-81-4550` (spaceID `11900388`, no `CA-xxx` name): p99 implied average power 45.6 kW, 1,130 sessions = 3.6% of Caltech sessions but 8.6% of kWh.
  Caltech data has 55 stations while acnportal's `caltech_acn()` has 54 ports, so this is the extra one. JPL has 52 stations = 52 ports, all <= 6.5 kW.
- Uncontrolled replay shortfall vs recorded kWh (midnight='drop'): Caltech 7.37% with that station, 1.19% without it; JPL 0.25%.
  (midnight='clip': Caltech 8.47%, JPL 0.37%.) In every case each vehicle receives exactly its clipped request (max error 4e-15).
- Not excluded yet. Excluding it is a data-definition change (default today: included). Needs the user's decision.

**Audit numbers (after dropping midnight crossers, midnight='drop' | 'clip'), weekdays only, DST days: 0 affected (DST changes fall on Sundays)**
- Caltech: 767 | 768 weekdays, of which Jun-Sep (real summer tariff) 285 | 285; sessions crossing midnight 1,984 (6.31% of all); sessions with `e_i` clipped 2,542 | 2,879; max concurrency 49 (< 54 ports).
- JPL: 702 | 703 weekdays, Jun-Sep 219 | 220; midnight crossers 488 (1.45%); `e_i` clipped 1,331 | 1,399; max concurrency 52 (= N, no vehicle ever lacked a port).
- Weekdays per year (drop): Caltech 2018 179, 2019 260, 2020 146, 2021 182. JPL 2018 82, 2019 260, 2020 184, 2021 176.
- Sessions per month at Caltech fall from about 2,400 (Aug-Oct 2018) to 1,309 (Nov 2018) and about 900 for 2019 (consistent with the Nov 2018 pricing change that
  Li & Sun mention); COVID collapse from Mar 2020 (381 sessions, then 5-70/month through Nov 2020). JPL is steadier (about 1,300-1,700/month until Feb 2020, 25-500 after).
- Candidate split periods the audit supports (user to choose): Caltech 2019-01 .. 2020-02 (about 14 months, one pricing regime, pre-COVID); JPL 2018-10 .. 2020-02.
  Post-shift test candidates: 2021 (Caltech 2021-01..2021-09, JPL same). Not chosen.

## 2026-10-04 (evening) user delegated the open decisions ("do what you think is best as an intermediate-advanced RL person")
The decisions below were MADE BY CLAUDE under that delegation. They are in `configs/base.yaml`; change the yaml (and tell me) to revisit.
- **Caltech fast station `2-39-81-4550` excluded** (not a 6.656 kW port; the design models 54 ports x 6.656 kW). Uncontrolled-replay shortfall 7.37% -> 1.19%.
- **Midnight-crossing sessions dropped** (6.3% Caltech, 1.45% JPL): clipping departures to day end would invent an earlier departure than the real one, which is
  exactly the quantity this study manipulates. Dropping is cleaner.
- **Tariff = summer curve on ALL weekdays** (option b). The real Jun-Sep days inside the stable period are only about 85, too few to train/validate/test an RL
  policy, and mixing real winter/summer curves would add a price shift that confounds the report-uncertainty effect we study. Disclose in the report as a controlled assumption.
- **DST**: moot (0 weekdays affected; DST changes happen on Sundays). Code still drops such days.
- **Splits (local dates, chronological, one pricing regime, pre-COVID)**: Caltech train 2019-01-01..2019-08-31 (174 days, 6,206 vehicles), val 2019-09-01..2019-10-31
  (44 days, 1,536 veh), test 2019-11-01..2020-02-29 (85 days, 2,720 veh), post-shift 2021-01-01..2021-09-30 (182 days, only 9 veh/day vs 34). JPL (E6 transfer) uses the
  same test/post windows (85 days 5,329 veh / 176 days 5,533 veh). Caltech sessions per month drop in Nov 2018 (pricing change) and collapse in Mar 2020 (COVID), hence these bounds.
- **Solver choice (open decision 7)**: decided in favour of a single-file PPO-Lagrangian written here (details further down once the compatibility check is run).
- **MPC (decision 8)**: will be done after Phase 1, because it reuses the LP.

### M1 (recorded driver reports): the premise of dropping it was wrong, so I measured it (descriptive only; design E1-E6 untouched)
- `scripts/analyze_recorded_reports.py` -> `results/recorded_report_error_{caltech,jpl}.csv`. Error = recorded requestedDeparture - true disconnect (hours).
  Caltech (16,365 sessions): mean |err| 3.15 h, median -0.34 h, 5%/95% quantiles -7.7/+7.5 h; only 35% within 1 h, 51% within 2 h; 59% said EARLIER than the real departure, 41% LATER.
  JPL (31,449): mean |err| 2.35 h, median -0.37 h, 5%/95% -6.4/+5.1 h; 39% within 1 h, 58% within 2 h; 61% earlier, 39% later.
  After COVID the bias flips to "later" (Caltech mean error from about -1.5 h in 2019 to about +3.8 h in 2021): real report behaviour itself shifts.
- Consequence for M2: uniform +-w has mean |err| = w/2, so the design's w in {0.5, 1, 2} h are far milder than reality (Caltech would need w of about 6 h).
- Added a SUPPLEMENTARY mode `M1r` (`report_model.py`): the recorded departure where one exists (Caltech 81-88%, JPL 97% of vehicles in the splits), otherwise an error
  resampled from the empirical error pool of the same data, seeded and reused across methods like the other modes. It will be reported separately from E1-E6,
  clearly labelled as an extra experiment. The numbering M0/M2/M3/M4 is untouched. THE TEAM MUST CORRECT the "0.2%" statement in Section X.A/X.D of the submitted text
  (it came from the first 500 sessions, all from April 2018, before the app collected the field).
- Code changes: `data.py` (`Day.rep_rec`, `exclude_stations`), new `config.py`, `configs/base.yaml`, test for M1r (78 tests passing).

### Calibration (TRAIN days only, M0) -> kappa = 0.20 chosen as the MAIN setting
- `scripts/calibrate.py` -> `results/calibration_kappa.csv`, `configs/derived_caltech_kappa0.2.yaml`; `scripts/calibrate_kappa_reports.py` -> `results/calibration_kappa_reports.csv`.
- P_peak (median daily peak if every vehicle charged at full rate on arrival) = 173.1 kW (all 54 ports would be 359.4 kW). e_ref = mean e_i = 8.359 kWh. 35.7 vehicles/day, 298 kWh requested/day.
- Offline-optimal unmet share U* by kappa: 0.15 -> 4.4% (the 2% budget is INFEASIBLE even with full information), 0.20 -> 0.56%, 0.25 -> 0.04%, >= 0.30 -> 0.
  With truthful reports EDF/LLF are near-optimal on U once kappa >= 0.25; the remaining gap is COST (rules about 7-10% above the offline optimum per kWh).
- Rule baselines under reports (U, train days): at kappa = 0.20, EDF 1.1% (M0), 1.6% (M2, w=2 h), 3.7% (M1r recorded); LLF 0.56% (M0), 0.9% (M2), 2.8% (M1r).
  Reports only start to matter when the cap is tight (kappa <= 0.25); real recorded errors hurt about 2-3x more than M2 with w = 2 h.
- Rule: main kappa = the tightest value at which the unmet budget is still achievable with full information => 0.20 (P_grid = 34.61 kW, binds in 66% of occupied steps).
  Looser/tighter variants for the design's "also looser and tighter kappa": 0.30 and 0.15 (0.15 stress case: d1 unachievable by construction).
- Derived constants at kappa = 0.20: d1 = 0.7133 episode-cost units (= 2% of requested energy per day / e_ref), d2 = 2.923 (mean of FCFS/EDF/LLF episode wear proxy sum_t c2_t,
  train, M0; taken literally as 1.0x the baselines), reward_scale = 4.78 $ (so an FCFS day returns about -10), offline M = $5/kWh.
- NOTE d2 is almost non-binding for a policy that charges below 0.8 P_port; can be tightened later (e.g. 0.5x the baselines) as an extra, not done.

### Constrained-PPO decision (open decision 7) and the Phase 1 framework
- Install dry run: OmniSafe resolves in this venv but pulls about 40 extra packages (safety-gymnasium, wandb, ...). To my knowledge its PPO-Lagrangian has ONE cost / multiplier; this
  design needs two (unmet energy, wear), plus a custom masked Deep Sets policy and a custom env adapter. I did NOT run it, only the install dry run. Decision: single-file PPO-Lagrangian
  written here (`src/safecharge/agents/ppo_lag.py`, about 170 lines). This is a deviation from the design wording "existing implementation of constrained PPO": flagged for the team.
- `agents/deepsets.py`: shared per-port encoder, masked mean-pool, shared head -> mean/log-std per port; critic = pooled read-out with 3 heads (reward, c1, c2). Implementation choices
  on top of the design (flagged): (a) Gaussian over a with u = sigmoid(a) instead of a Gaussian directly over u in [0,1]; (b) the occupied-port COUNT is appended to the pooled context
  (mean-pool alone cannot see how many vehicles compete; config switch `use_count: false` gives the literal design); (c) empty ports masked out of the log-prob and entropy.
- PPO-Lagrangian details: separate GAE for reward and the two costs (gamma 0.99, lambda 0.95), advantages normalised per signal by batch std, combined as (A_R - l1 A_C1 - l2 A_C2)/(1 + l1 + l2),
  dual ascent l_k <- max(0, l_k + alpha (J_k/d_k - 1)) on the EMA-smoothed UNDISCOUNTED episode cost J_k (alpha = 0.05, relative violation so alpha does not depend on cost units).
  FLAG: the design writes J_C with gamma-discounting; using undiscounted episode cost for the dual step is common in PPO-Lagrangian code and makes "d1 = 2% of requested energy" mean exactly that.
  Critic targets use running per-head scales (the CLAUDE.md "running normalisers"); reward additionally scaled by `reward_scale`.
- Tests added: Deep Sets permutation equivariance/invariance, masking of empty ports, log-prob vs torch Normal, GAE by hand, dual ascent. 85 tests pass.
- Scripts: `scripts/train.py` (resumable, CSV log, best/last checkpoints; best = feasible-on-validation first, then lowest cost/kWh), `scripts/run_experiment.py` (same columns for every method,
  bootstrap CIs over test days, per-day gap to the offline optimum, asserts the LP lower bound on every day for rules and RL).
- Bugs found and fixed while building: dual-step cost axis (summed over wrong axis); LP metrics lacked stay-duration columns. Both caught by running, not by silent failure.
- Rules on validation, kappa 0.2, M0 (44 days): FCFS $/kWh 0.1772 U 5.6% | EDF 0.1790, 1.5% | LLF 0.1798, 0.48% | OPT 0.1629, 0.47%. (exploration data, not final)

### Phase 1 exploration (seed 0, VALIDATION split only; none of these are final numbers; test split not touched)
- Run A (first config, 600 updates, cold start, gamma 0.99): val U 3.7%, 0.1845 $/kWh (rules: EDF 1.5% / 0.1790, LLF 0.48% / 0.1798, FCFS 5.6% / 0.1772) -> worse than the rules. lambda_1 wound up to 89
  (relative-violation dual step unbounded; it would take about 1800 updates to unwind) => CHANGED `dual_step`: relative violation clipped to [-1, 1] and lambda capped (lam_max = 10); tests added.
- Runs v1/v2/v3 (dual fix; +gamma 0.999 / GAE 0.97; +32 envs, lr 1e-3), 400 updates, cold start: all about val U 4-5% at update 360-400, cost 0.186-0.188. Only about 1.4x faster than A.
  Cold start is slow because the unmet-energy cost arrives only at departure and the uniform-cut projection needs a ranking over ports the mean-pooled context only sees roughly.
- Added (flagged: NOT in the submitted design, reported separately): **behaviour-cloning warm start** from LLF (`PPOLag.bc_pretrain`, regress the actor mean onto logit(u_LLF) on LLF's own train-day trajectories)
  plus a 30-update critic warm-up (actor and lambda frozen). BC alone with 30 epochs: val U 2.7%, 0.1841 $/kWh; 150 epochs: U 1.6%, 0.1813 $/kWh (LLF 0.48% / 0.1798), so the Deep Sets policy can represent most of LLF.
  The first BC+RL run (30 epochs) was initially misread as a train/validation gap; a proper same-pipeline check (train 44 random days vs val, deterministic vs stochastic) showed none (single-batch logged J1 values are very noisy: 0.3-2.3).
- Added the design's listed ablations behind config switches (default unchanged = design): `pool: meanmax | attn` in `deepsets.py` (self-attention across occupied ports = the design's "possible later extension"), and `scheme: priority` projection (E5).
  All pooling variants pass the permutation/mask tests (88+ tests pass).
- Added **MPC baseline** (`src/safecharge/mpc.py`, receding-horizon LP, declared deadlines only, same penalty M; tests: single-vehicle day equals the offline optimum, offline bound holds). Exploration on VAL, kappa 0.2:
  M0: MPC 0.1719 $/kWh (5.8% above OPT), U 0.81%, 0.78 ms/step. Under recorded reports (M1r): MPC U jumps to 17.2% (EDF 4.2%, LLF 3.4%) because it trusts declared deadlines and postpones charging.
  Not final (val split, one report draw).
- Process note: pgrep-based waiters matched themselves and never exited; fixed with the `[s]cripts` pattern. Background waiter time limits kill long waits, so long jobs are polled.

### 2026-10-04 (night) candidate comparison, final configuration, Phase 3/4 tooling
- Validation comparison, seed 0, uniform-cut unless noted (U = unmet share, $/kWh; feasible = J1 <= 0.713 and J2 <= 2.92):
  cold start 2000 upd: U 2.0-2.5%, 0.190-0.200 (dominated by every rule) | BC-LLF 800 upd (c3): feasible early, U 1.6-2.3%, 0.180, unstable late | BC-LLF + priority projection (c4): U 1.3-1.5%, 0.180-0.184, stays feasible
  | BC-LLF + 48 envs (d1): U 1.4-1.9%, 0.180-0.181 | + max-pool context (d2): U 2.5-2.9%, 0.178 | + self-attention (d3): U 1.4-2.7%, 0.188-0.200 (worse cost) | BC-MPC (d4): U 1.4%, 0.182 (feasible after about 500 updates).
  None found the price-shifting of MPC (0.172) / OPT (0.163): the learned policies sit at about LLF level in cost (0.180) with higher unmet energy (LLF 0.48%). Treat as a finding, not a failure to hide.
- FINAL main config `configs/e1.yaml` (selected on validation, seed 0): design-faithful architecture (mean-pool, uniform cut) + BC warm start + gamma 0.999 + dual clipping/cap + lr decay. `configs/e1_cold.yaml` = same without BC (control).
  5 seeds x 1500 updates, at most 3 in parallel. d1 (the exploratory run of this config for seed 0) was stopped at update 800; the final seed 0 is a fresh run.
- INCIDENT (caught): a stale `results/runs/e1_m0/seed0` from my first cold-start run was auto-resumed ("resumed at update 600") by the new run of the same name; detected from the log, directory deleted, all seeds restarted clean.
- JPL transfer kappa by the same rule (`scripts/calibrate_transfer_site.py`, JPL train days, P_peak 339.5 kW, 64 veh/day, 935 kWh/day): offline U* = 13.7% at kappa 0.20 (the Caltech kappa is far too tight for JPL), 0.30% at 0.28 -> JPL kappa = 0.28 (P_grid 95 kW).
  `run_experiment.py` / `run_paired_replay.py` now recalibrate P_peak from the transfer site's own train days.
- Phase 3/4 tooling added: env option `curtail` (E4 mitigation: no power at t >= ceil(reported departure); tests incl. "curtailment never rewards under-reporting under EDF"), `scripts/run_paired_replay.py` (E3: paired replay,
  focal vehicles with stays >= 2 h, bootstrap over days), `scripts/run_eval_suite.sh` (E1/E2/E3/E6 for one config), E4 configs `e4_random` (mixture incl. M1r), `e4_curtail`, `e4_random_curtail` (mixture weights are my choice).
- 108 tests pass.

### PRELIMINARY E1 (test split, M0, kappa 0.2, 85 days, RL = seeds 0-2 only; superseded by the 5-seed run) `scripts/run_experiment.py ... --seeds 0,1,2`
- $/kWh (gap to OPT) | U | miss rate | W (high-power ratio): FCFS 0.1708 (10.2%) | 4.79% | 9.1% | 0.78; EDF 0.1732 (11.7%) | 1.25% | 1.2% | 0.77; LLF 0.1738 (12.1%) | 0.43% | 1.7% | 0.77;
  MPC 0.1649 (5.5%) | 0.67% | 2.5% | 0.77; OPT 0.1564 | 0.43% | 0.6% | 0.79; PPOLag 0.1731 (11.7%) | 1.29% | 7.0% | 0.17. Grid violations 0 for every method (projection works).
- Reading: the learned policy matches EDF on cost and unmet energy, is about 0.4% cheaper than LLF but leaves about 3x LLF's unmet energy, and has a much higher share of vehicles below 95% (7.0% vs 1.2-1.7%).
  Its distinctive property is gentle charging (W 0.17 vs 0.77, wear proxy 1.50 vs 2.4-2.5 per day) at no cost penalty. MPC with truthful reports is the best on cost. Decision time: RL 0.26 ms/step, MPC 0.93 ms/step, rules 0.09 ms.
- Not yet final: seeds 3-4 still training; results will be regenerated with `scripts/run_eval_suite.sh configs/e1.yaml e1_m0`.

### 2026-10-05 FINAL E1/E2/E3/E6 (5 seeds, test split) -> see `results/FINDINGS_phase1-3.md` and `results/tables/`
- Training: `configs/e1.yaml`, seeds 0-4 x 1500 updates (`results/runs/e1_m0/`, gitignored). Evaluation: `scripts/run_eval_suite.sh configs/e1.yaml e1_m0` (log `results/runs/e1_m0/suite.log`), tables via `scripts/make_tables.py e1_m0`.
- Headline (all in the findings file): E1 the learned policy ties EDF on cost/unmet energy, does not beat the rules, MPC best on cost; policy charges gently (W 0.17 vs 0.77). E2 degradation: learned policy 2.6x worse under M2 w=2 h,
  rules 1.2-1.6x, MPC 30x; under recorded reports (M1r) U: PPOLag 7.4%, EDF 3.8%, LLF 3.0%, MPC 19.6%. E3: learned policy gained by under-reporting in 93-97% of paired replays (+0.84..1.50% of e_i at kappa 0.2), EDF 1.5-1.9% of cases (+0.22..0.57%),
  gains larger at tight kappa 0.15. E6: post-shift fine under M0, JPL transfer fails (U 28.6%; also with a non-binding cap).
- Verification done on these runs: offline LP lower bound asserted on every day for FCFS/EDF/LLF/MPC and each RL seed; 0 grid violations everywhere; same days and report draws for every method (report seed 12345).
- Diagnostic runs not kept as results: JPL with kappa 1.0 (policy still leaves 23.6% unmet).
- E4 training started: 3 configs (`e4_random`, `e4_curtail`, `e4_random_curtail`) x seeds 0-2 (NOT 5; time), 3 in parallel. Results to be added when done.
- Caveats written down in the findings file (design additions, tuning on validation with seed 0, evaluation-only kappa transfers, M1r supplementary).

### 2026-10-05 E4 status (end of this working session)
- E4 part 1 done (no training): standard policy + rules + MPC evaluated in the curtail environment (`results/e1_m0_E4_curtail_*`, findings appended to `results/FINDINGS_phase1-3.md`):
  curtailment makes under-reporting a loss for every method (-0.6..-1.8% at s <= 1 h, about -9% at 2 h) but roughly doubles unmet energy for honest-but-wrong reports (e.g. M2 w=2 h: EDF 1.5% -> 6.2%).
- E4 part 2 (policies TRAINED with randomised reports / in the curtail env): `e4_random`, `e4_curtail`, `e4_random_curtail`, seeds 0-2, launched as 9 runs, 3 in parallel; at the time of writing wave 1 (seed 0) is at update 1000-1400 of 1500.
  Each run takes about 35-40 min on this laptop (browsers compete for CPU). When all 9 are `done`: run `scripts/run_e4_suite.sh configs/e4_random.yaml`, `... configs/e4_curtail.yaml --curtail`, `... configs/e4_random_curtail.yaml --curtail`, then add the findings.
  Validation snapshots (seed 0, not results): e4_random U 2.3-2.7% / 0.179-0.186; e4_curtail U 1.5-2.0% / 0.180-0.183; e4_random_curtail U 1.4-1.8% / 0.181.
- NOT done: E5 ablation (priority projection, attention/max-pool, unprojected RL with soft grid cost): the switches exist (`scheme: priority`, `pool: meanmax|attn`) and validation-only exploratory runs exist (see above), but no 5-seed test-split evaluation; unprojected RL (soft grid-violation cost) is not implemented. SAC-Lagrangian not done.
- Things the team must handle in the report (not code): (1) the "0.2% requestedDeparture" statement in Section X.A/X.D is wrong for the full data (52% Caltech, 94% JPL); (2) RL additions not in the submitted design are listed in `results/FINDINGS_phase1-3.md` caveats; (3) tariff applied to all weekdays, fast station excluded, midnight sessions dropped, kappa 0.20 / JPL 0.28 are chosen constants, not in the submitted design.

### 2026-10-05 00:45 IST E4 training paused on the user's request (laptop going to sleep); nothing lost
- Stopped the xargs queue and the 3 running trainers. All 9 `last.pt` files load. DONE (1500/1500): e4_random s0,s1; e4_curtail s0,s1; e4_random_curtail s0,s1. TO RESUME: e4_curtail s2 (at 1240), e4_random s2 (1280), e4_random_curtail s2 (760).
- Resume command (re-running a finished run does nothing and exits at once, so the whole list is safe to re-run; each resumes from its `last.pt`):
  `cd safecharge_rl; set -a; source .env; set +a` is NOT needed (training uses no token), then
  `caffeinate -i bash -c 'for c in e4_curtail e4_random e4_random_curtail; do echo "$c 2"; done | xargs -P 3 -L 1 sh -c ".venv/bin/python -u scripts/train.py configs/\$0.yaml --seed \$1 >> results/runs/\$0/train_seed\$1.log 2>&1"'`
- Resume caveats (honest): checkpoints are written every 20 updates, so up to 20 updates of each interrupted run are redone; the NumPy RNG state is restored but the torch RNG is not, so a resumed run is not bit-identical to an uninterrupted one
  (same config and seed, statistically equivalent); `log.csv` may hold up to 20 duplicated update rows around the resume point.
- After they finish: `scripts/run_e4_suite.sh configs/e4_random.yaml`, `... configs/e4_curtail.yaml --curtail`, `... configs/e4_random_curtail.yaml --curtail`; then add the E4 findings to `results/FINDINGS_phase1-3.md`.
