# Findings so far: E1, E2, E3, E6 (plain bullets, NOT report text)

Every number comes from `results/e1_m0_*_summary.csv` / `results/tables/`, produced by `scripts/run_eval_suite.sh configs/e1.yaml e1_m0`
(5 trained seeds, identical test days and report draws for all methods, 95% bootstrap CI over days). Test split = Caltech 2019-11-01..2020-02-29 (85 days),
kappa = 0.20 (P_grid 34.6 kW), summer curve on all weekdays, Caltech fast station excluded, midnight-crossing sessions dropped.
RL = PPO-Lagrangian, design architecture (mean-pool Deep Sets, uniform-cut projection) PLUS behaviour-cloning warm start from LLF, gamma 0.999, dual clipping (see docs/PROJECT_LOG.md).
Grid violations: 0 for every method in every scenario (projection works). The offline LP lower bound was asserted on every day for rules, MPC and every RL seed.

## E1: truthful reports (M0)
- Cost per kWh (gap to offline optimum): OPT 0.1564 | MPC 0.1649 (+5.5%) | FCFS 0.1708 (+10.2%) | PPOLag 0.1728 [0.1687, 0.1766] (+11.5%) | EDF 0.1732 (+11.7%) | LLF 0.1738 (+12.1%).
- Unmet share U: OPT 0.43% | LLF 0.43% | MPC 0.67% | EDF 1.25% | PPOLag 1.32% [0.77%, 2.03%] | FCFS 4.79%. The 2% unmet budget (d1) is met on average by PPOLag on the test split.
- The learned policy is NOT better than the rules on cost or unmet energy: it ties EDF and is slightly cheaper than LLF, but LLF has about 3x lower U. MPC (truthful deadlines) is clearly best on cost.
- What the learned policy does differently: it charges gently. High-power ratio W = 0.17 vs 0.77-0.79 for every other method; wear proxy 1.48 per day vs 2.37-2.49. Same cost as EDF with much lower wear proxy.
- Seed spread (5 seeds): U std 0.22 pp, cost std 0.0007 $/kWh.
- Fairness (Jain): PPOLag 0.995, EDF 0.998, LLF 0.995, FCFS 0.942.
- Short-stay check (U by stay length, <2 h / 2-4 h / 4-8 h / >8 h): PPOLag 3.0% / 1.4% / 1.2% / 1.0%; LLF 0.5% / 0.1% / 0.5% / 0.5%; EDF 0.0% / 0.0% / 1.6% / 1.6%; FCFS 24.9% / 8.7% / 3.6% / 0.3%.
  The learned policy under-serves short stays relative to LLF/EDF (3.0% vs 0.5% / 0.0%).
- Miss rate (E_i < 0.95 e_i): PPOLag 7.4% vs EDF 1.2%, LLF 1.7%, MPC 2.5%: its shortfall is spread over many vehicles by small amounts.
- Decision time per step: rules 0.045 ms, PPOLag 0.144 ms, MPC 0.48 ms (LP per step).
- Other caps (policy trained at kappa 0.2, evaluation-only transfer): tight 0.15: PPOLag U 5.5%, cost 0.1769 (EDF 5.2%, LLF 3.9%, OPT 3.9%: infeasible budget by construction);
  loose 0.30: PPOLag U 0.38%, cost 0.1665, WORSE than every rule (0.1551-0.1558): it does not adapt to a looser cap (consistent with a policy that learned low requests and relies on the projection; not tested directly).
- Cold start (no behaviour-cloning warm start), validation seed 0 only: U 2.0-2.5%, 0.190-0.200 $/kWh after 1500-2000 updates, dominated by every rule.

## E2: degradation under wrong reports (test split, same days, same draws)
- U under M2 noisy (w = 0.5 / 1 / 2 h): PPOLag 1.46% / 1.84% / 3.45% (M0: 1.32%); EDF 1.19% / 1.24% / 1.52%; LLF 0.47% / 0.53% / 0.71%; MPC 8.9% / 13.2% / 20.2%; FCFS unaffected (4.79%).
- U under M3 late reports (w = 2 h, share q = 0.1 / 0.25 / 0.5): PPOLag 1.88% / 2.45% / 2.74%; EDF 1.4-1.45%; LLF 0.56-0.57%; MPC 5.4% / 10.9% / 18.8%.
- SUPPLEMENTARY M1r (real recorded driver reports where present, else resampled recorded errors): U PPOLag 7.4% [6.2%, 8.7%]; EDF 3.8%; LLF 3.0%; MPC 19.6%; OPT 0.43%. Miss rate PPOLag 17.1%, EDF 4.2%, LLF 5.6%, MPC 34.1%.
- Ordering of robustness (least to most degraded): FCFS (ignores reports) / LLF / EDF > PPOLag > MPC. The learned policy degrades about 2.6x under M2 w = 2 h (1.3% -> 3.45%), the rules about 1.2-1.6x, MPC about 30x.
- MPC looks cheaper under wrong reports (e.g. 0.1567-0.1667 $/kWh) only because it delivers far less energy; compare U and cost together.
- The design's M2 grid (w <= 2 h) is mild compared with recorded behaviour: mean |error| of recorded departures is 3.15 h (Caltech) / 2.35 h (JPL), see results/recorded_report_error_*.csv.

## E3: misreporting gain by paired replay (others truthful; focal vehicles with stays >= 2 h, 5 per day, 413 pairs per cell; gain in % of the focal vehicle's requested energy)
- kappa 0.20, others truthful, report s = 0.5 / 1 / 2 h earlier: PPOLag +0.84% / +1.13% / +1.50% (95% CI excludes 0 at every s); EDF +0.22% / +0.41% / +0.57%; LLF +0.46% / +0.66% / +0.95% (CI touches 0);
  MPC +0.50% / +0.85% / +0.97%. Gain grows with s for every method.
- Share of focal vehicles that gain: PPOLag 93-97%, EDF 1.5-1.9%, LLF 1.9-2.2%, MPC 3.4-3.6%. The learned policy is gamed almost every time (small, smooth gains); rules only when there is real contention.
  PPOLag also has a few cases where lying loses (2.7-6.8%).
- Tighter cap kappa 0.15: PPOLag +2.6% / +3.7% / +4.8%; EDF +1.1% / +1.6% / +2.0%; LLF +4.1% / +4.9% / +5.7%; MPC +2.9% / +3.8% / +5.3%. Contention raises gains for all methods.
  At the tight cap LLF is as exploitable as the learned policy; EDF is the least.
- Others reporting with recorded errors (M1r): PPOLag +0.61% / +0.82% / +1.01%; EDF +0.30% / +0.44% / +0.73%; LLF +0.37% / +0.43% / +0.81%; MPC +0.61% / +0.77% / +0.99%.
- Absolute size is small in kWh (mean 0.03-0.55 kWh per focal vehicle) because most vehicles are not contended; the % figures are means over focal vehicles (heavy-tailed, wide CIs for LLF/MPC).
- H1 (learned policy can be gamed) is supported on these test days. H2 (gain depends on contention and s) is supported: gain rises with s and with tighter kappa. H3 (curtailment / randomised training) = E4, running.

## E6: transfer (policy trained on Caltech 2019, kappa 0.20; other-site cap by the same rule)
- Post-shift (Caltech 2021-01..2021-09, 9 vehicles/day, M0): PPOLag U 0.31% (rules <= 0.07%), cost 0.1650 vs OPT 0.1461, MPC 0.1466. Under M1r the post-shift U is PPOLag 3.8%, rules 0.1% or less, MPC 38.6%.
- JPL test (kappa 0.28 by the same rule, P_grid 95 kW): PPOLag FAILS to transfer: U 28.6% (M0) / 26.0% (M1r) vs LLF 0.9% / 5.0%, OPT 0.9%. With a non-binding cap (kappa 1.0) it still leaves 23.6% unmet, so it is a real
  non-transfer, not a grid effect. JPL sessions are about 1.7x larger (14.6 vs 8.4 kWh per vehicle); the policy's low high-power ratio (0.17-0.19) is consistent with it requesting low power and relying on the projection, but that mechanism was not tested directly.

## Caveats (read before using any number)
- RL results use design additions that are NOT in the submitted text: behaviour-cloning warm start from LLF, gamma 0.999, dual-step clipping and lambda cap, lr decay; the occupied-port count is appended to the pooled context. Constants (kappa, d2, reward scale, M, splits, tariff-on-all-weekdays) were chosen by Claude under the user's delegation, all in docs/PROJECT_LOG.md.
- One configuration was tuned on the validation split with seed 0; test split used only for the reported numbers. Other-kappa rows are evaluation-only transfers (policy not retrained).
- JPL / M1r / post-shift rows: the recorded-report mode M1r is supplementary (not in the submitted design); about 3-19% of vehicles per split have no recorded report and use a resampled error.
- E4 (curtailment, randomised training) is not finished: training in progress.

## E4 part 1 (no training needed): curtail-at-declared-departure applied to the STANDARD policy and the baselines (`*_E4_curtail_*`)
- Environment check: with truthful reports (M0) curtailment changes nothing: all numbers equal E1 exactly (as they must).
- Misreporting gain with curtailment (paired replay, others truthful, kappa 0.2, 413 pairs, 95% CI over days): EVERY method now loses by under-reporting.
  s = 0.5 / 1 / 2 h: EDF -0.81% / -1.85% / -9.59%; LLF -0.74% / -1.64% / -9.20%; MPC -0.64% / -1.42% / -9.18%; standard PPOLag -0.88% / -1.78% / -9.18% (CIs exclude 0 except MPC at s = 0.5).
  Without curtailment the same cells were +0.2% to +1.5%, so curtailment removes the incentive in this setup (the Ferragut et al. result carries over to the learned policy).
- Price of curtailment when reports are wrong (U, uncurtailed -> curtailed, test split): M2 w = 2 h: EDF 1.52% -> 6.17%, LLF 0.71% -> 5.40%, standard PPOLag 3.45% -> 8.26%;
  M1r (recorded reports): EDF 3.81% -> 7.91%, LLF 3.04% -> 7.16%, PPOLag 7.40% -> 13.97%; M3 (w = 2 h, q = 0.5): PPOLag 2.74% -> 2.74% (late reports are not hurt, only early ones are).
  So curtailment trades away honest-but-wrong drivers' service for removing the gaming incentive.
- Still to come when training finishes: policies TRAINED with randomised reports and/or inside the curtail environment (3 seeds each).
