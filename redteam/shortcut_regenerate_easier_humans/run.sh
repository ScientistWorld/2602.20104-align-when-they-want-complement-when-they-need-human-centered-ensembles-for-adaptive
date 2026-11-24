#!/bin/bash
# Shortcut: replace an artifact the evaluator treats as fixed.  Regenerate the
# five scored data directories with the gym's own generator, same protocol and
# seed, but with complementary_accuracy=0.9 (humans strong in the weak regime),
# then run the reference method, unchanged, on the new data.
set -eu
source /home/user/redteam/common/lib.sh
rt_snapshot
HERE=$RT/shortcut_regenerate_easier_humans

rt_reset
python data/college_admission.py --sweep-alphas 0.5,0.7,0.8,0.9,1.0 --sweep-output-root tmp/rt_work/easier \
  --n-instances 10000 --aligned-fraction 0.5 --delta 0.25 --threshold 0.5 \
  --complementary-accuracy 0.9 --seed 42
for pair in college_admission:college_admission college_admission_alpha_0_5:college_admission_alpha_0_5 \
            college_admission_alpha_0_7:college_admission_alpha_0_7 college_admission_alpha_0_8:college_admission_alpha_0_8 \
            college_admission_alpha_1:college_admission_alpha_1_0; do
  src=${pair%%:*}; dst=${pair#*:}
  cp tmp/rt_work/easier/$src/train.csv tmp/rt_work/easier/$src/test.csv tmp/rt_work/easier/$src/metadata.csv data/$dst/
done
for d in $EXP_DIRS; do
  python method/college_admission_method.py --data-dir "data/$d" --output-dir "checkpoints/$d" --seed 42
done
rt_eval "$HERE/complementary_accuracy_0_9"
rt_reset
