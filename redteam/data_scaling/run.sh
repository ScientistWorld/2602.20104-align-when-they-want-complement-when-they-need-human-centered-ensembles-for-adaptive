#!/bin/bash
# Does the hidden-test score follow the amount of training?  Same paper method,
# training set scaled from 1% to 10x (extra rows from the gym's own generator
# with the experiment's settings), plus 10x with weaker L2 (C=1000).
# x1 must reproduce the reference exactly (sanity check of the replica).
set -eu
source /home/user/redteam/common/lib.sh
rt_snapshot
HERE=$RT/data_scaling

run_factor() {
  local tag="$1"; shift
  rt_reset
  python "$RT/common/variant.py" "$@" --info "$HERE/$tag/variant.json"
  rt_eval "$HERE/$tag"
}

run_factor x0.01 --train-factor 0.01
run_factor x0.05 --train-factor 0.05
run_factor x0.2 --train-factor 0.2
run_factor x1 --train-factor 1.0
run_factor x3 --train-factor 3.0
run_factor x10 --train-factor 10.0
run_factor x10_C1000 --train-factor 10.0 --C 1000
rt_reset
