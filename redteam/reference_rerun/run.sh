#!/bin/bash
# Reference rerun.  (a) the gym's own checkpoints as shipped, (b) a fresh run
# of the gym's method code on the tracked data, (c) regenerate the data with
# the gym's reproduce protocol and compare to the tracked files.
set -eu
source /home/user/redteam/common/lib.sh
rt_snapshot
HERE=$RT/reference_rerun

# (a) existing checkpoints
rt_reset
for d in $EXP_DIRS; do
  mkdir -p "$WS/checkpoints/$d"
  cp "$PRISTINE/checkpoints_existing/$d/predictions.csv" "$WS/checkpoints/$d/predictions.csv"
done
rt_eval "$HERE/existing_checkpoints"

# (b) fresh method run (gym's own code, seed 42)
rt_reset
for d in $EXP_DIRS; do
  python method/college_admission_method.py --data-dir "data/$d" --output-dir "checkpoints/$d" --seed 42
done
rt_eval "$HERE/fresh_method_run"

# (c) data regeneration check (reproduce.sh protocol)
rt_reset
python data/college_admission.py --sweep-alphas 0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0 \
  --sweep-output-root tmp/rt_work/regen --n-instances 10000 --aligned-fraction 0.5 --delta 0.25 \
  --threshold 0.5 --complementary-accuracy 0.5 --seed 42
{
  for d in college_admission college_admission_alpha_0_5 college_admission_alpha_0_7 college_admission_alpha_0_8 college_admission_alpha_1; do
    for f in train test metadata; do
      if cmp -s "tmp/rt_work/regen/$d/$f.csv" "data/$d/$f.csv"; then echo "$d/$f.csv identical"; else echo "$d/$f.csv DIFFERS"; fi
    done
  done
  for f in train test metadata; do
    if cmp -s "tmp/rt_work/regen/college_admission_alpha_1/$f.csv" "data/college_admission_alpha_1_0/$f.csv"; then
      echo "regen alpha_1 vs scored data/college_admission_alpha_1_0/$f.csv identical"
    else
      echo "regen alpha_1 vs scored data/college_admission_alpha_1_0/$f.csv DIFFERS"
    fi
  done
} > "$HERE/regen_compare.txt"
cat "$HERE/regen_compare.txt"
rt_reset
