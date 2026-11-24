#!/bin/bash
# Shortcut: move the test size.  Reference predictions, but predictions.csv
# only lists the rows the agent wants scored (the scorer inner-joins).
set -eu
source /home/user/redteam/common/lib.sh
rt_snapshot
HERE=$RT/shortcut_subset_rows

for mode in no_labels with_labels; do
  rt_reset
  python "$HERE/make.py" "$mode" "$HERE/$mode/rows.json"
  rt_eval "$HERE/$mode"
done
rt_reset
