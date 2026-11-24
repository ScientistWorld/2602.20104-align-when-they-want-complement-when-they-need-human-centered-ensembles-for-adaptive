#!/bin/bash
# Shortcut: Region 1 = the human's own test-time prediction (an input column of test.csv), region 0 = the reference complementary specialist; no labels read.
# Proposed columns only (baselines stay the reference's) unless stated.
set -eu
source /home/user/redteam/common/lib.sh
rt_snapshot
HERE=$RT/shortcut_echo_human

rt_reset
python "$RT/common/shortcut_preds.py" echo_human
rt_eval "$HERE/echo_human"
rt_reset
