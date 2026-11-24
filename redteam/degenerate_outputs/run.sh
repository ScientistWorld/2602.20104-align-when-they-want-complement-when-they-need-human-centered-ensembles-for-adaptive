#!/bin/bash
# Degenerate outputs in the proposed columns (baselines = reference):
# constant 0 (no route), copy of the human's own input prediction (route =
# region), impossible class 2 with route 7, and NaN.
set -eu
source /home/user/redteam/common/lib.sh
rt_snapshot
HERE=$RT/degenerate_outputs

for mode in constant_zero copy_human_input impossible_values nan_values; do
  rt_reset
  python "$RT/common/shortcut_preds.py" "$mode"
  rt_eval "$HERE/$mode"
done
rt_reset
