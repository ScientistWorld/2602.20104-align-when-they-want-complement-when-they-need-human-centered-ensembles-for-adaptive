#!/bin/bash
# Seed spread of the reference method on the hidden test.
# (a) the method's own --seed (42, 43, 44): logistic regression with lbfgs is
#     deterministic, so this is expected to be zero spread.
# (b) bootstrap-resampled training set (seeds 1, 2, 3): the real run-to-run
#     variation of the reference training procedure.
set -eu
source /home/user/redteam/common/lib.sh
rt_snapshot
HERE=$RT/seed_spread

for s in 42 43 44; do
  rt_reset
  for d in $EXP_DIRS; do
    python method/college_admission_method.py --data-dir "data/$d" --output-dir "checkpoints/$d" --seed "$s"
  done
  rt_eval "$HERE/method_seed_$s"
done

for s in 1 2 3; do
  rt_reset
  python "$RT/common/variant.py" --bootstrap-seed "$s" --info "$HERE/bootstrap_seed_$s/variant.json"
  rt_eval "$HERE/bootstrap_seed_$s"
done
rt_reset
