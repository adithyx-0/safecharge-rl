#!/bin/bash
# E4 evaluation for one trained variant. Usage: scripts/run_e4_suite.sh configs/e4_random.yaml [--curtail]
# Rules/MPC are re-evaluated in the same environment (curtail or not) so each table compares like with like. Test split only. Seeds: all found (3).
set -e
C=$1; shift
CURT="$@"                       # "" or "--curtail"
P=$(basename "$C" .yaml)
PY=.venv/bin/python
R="$PY scripts/run_experiment.py $C $CURT --split test"
$R --mode M0 --tag ${P}_E4_test_M0
$R --mode M2 --w 2 --tag ${P}_E4_test_M2_w2
$R --mode M3 --w 2 --q 0.5 --tag ${P}_E4_test_M3_w2_q0.5
$R --mode M1r --tag ${P}_E4_test_M1r
PR="$PY scripts/run_paired_replay.py $C $CURT --split test --focal 5"
$PR --bg M0 --tag ${P}_E4_paired_bgM0
$PR --bg M1r --tag ${P}_E4_paired_bgM1r
echo "e4 suite done: $P $CURT"
