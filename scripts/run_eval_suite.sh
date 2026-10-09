#!/bin/bash
# Evaluation suite for one trained configuration. Usage: scripts/run_eval_suite.sh configs/e1.yaml [prefix]
# Runs E1 (M0 + looser/tighter kappa), E2 (M2/M3/M1r), E6 (post-shift, JPL), E3 (paired replay). Everything uses the TEST (or post) split only.
# Policies are trained at kappa 0.2; other-kappa rows are evaluation-only transfers. Needs the venv python.
set -e
C=${1:-configs/e1.yaml}
P=${2:-$(basename "$C" .yaml)}
PY=.venv/bin/python
R="$PY scripts/run_experiment.py $C"
# E1
$R --split test --mode M0 --tag ${P}_E1_test_M0
$R --split test --mode M0 --kappa 0.15 --tag ${P}_E1_test_M0_k0.15
$R --split test --mode M0 --kappa 0.30 --tag ${P}_E1_test_M0_k0.30
# E2: noisy and late reports, and recorded reports (supplementary M1r)
for w in 0.5 1 2; do $R --split test --mode M2 --w $w --tag ${P}_E2_test_M2_w$w; done
for q in 0.1 0.25 0.5; do $R --split test --mode M3 --w 2 --q $q --tag ${P}_E2_test_M3_w2_q$q; done
$R --split test --mode M1r --tag ${P}_E2_test_M1r
# E6: post-shift period (Caltech) and cross-site (JPL, kappa 0.28 by the same rule)
$R --split post --mode M0 --tag ${P}_E6_post_M0
$R --split post --mode M1r --tag ${P}_E6_post_M1r
$R --split test --site jpl --kappa 0.28 --mode M0 --tag ${P}_E6_jpl_test_M0
$R --split test --site jpl --kappa 0.28 --mode M1r --tag ${P}_E6_jpl_test_M1r
# E3: misreporting gain, others truthful and others with recorded errors
PR="$PY scripts/run_paired_replay.py $C --split test --focal 5"
$PR --bg M0 --tag ${P}_E3_paired_bgM0
$PR --bg M1r --tag ${P}_E3_paired_bgM1r
$PR --bg M0 --kappa 0.15 --tag ${P}_E3_paired_bgM0_k0.15
echo "suite done: $P"
