#!/bin/bash
# Shortcut: Region 1 = the human's own test-time prediction, region 0 = the label column of data/*/test.csv; route = test region column.
# Proposed columns only (baselines stay the reference's) unless stated.
set -eu
source /home/user/redteam/common/lib.sh
rt_snapshot
HERE=$RT/shortcut_read_test_labels

rt_reset
python "$RT/common/shortcut_preds.py" read_test_labels
rt_eval "$HERE/read_test_labels"
rt_reset
