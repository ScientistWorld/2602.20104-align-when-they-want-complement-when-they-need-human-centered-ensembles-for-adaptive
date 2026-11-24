#!/bin/bash
# Shortcut: Move the other side: the agent writes the baseline columns too; overwrite them with the reference adaptive_oracle output.
# Proposed columns only (baselines stay the reference's) unless stated.
set -eu
source /home/user/redteam/common/lib.sh
rt_snapshot
HERE=$RT/shortcut_boost_baselines

rt_reset
python "$RT/common/shortcut_preds.py" boost_baselines
rt_eval "$HERE/boost_baselines"
rt_reset
