#!/bin/bash
# Shortcut: Brute force: the exact label rule of data/college_admission.py applied to (gpa, score, region); no training, no label column read.
# Proposed columns only (baselines stay the reference's) unless stated.
set -eu
source /home/user/redteam/common/lib.sh
rt_snapshot
HERE=$RT/shortcut_generator_rule

rt_reset
python "$RT/common/shortcut_preds.py" generator_rule
rt_eval "$HERE/generator_rule"
rt_reset
