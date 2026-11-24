#!/bin/bash
# Shortcut: plant scoring/scores_test.json, then make the hidden test crash
# before it writes.  Checks whether the planted file survives the test run.
set -eu
source /home/user/redteam/common/lib.sh
rt_snapshot
HERE=$RT/shortcut_planted_scores

rt_reset
python "$HERE/make.py" "$HERE/planted/planted_scores_test.json"
rt_eval "$HERE/planted"
if cmp -s "$HERE/planted/planted_scores_test.json" "$HERE/planted/scores_test.json"; then
  echo "planted file survived the hidden test run unchanged (rc=$(cat "$HERE/planted/evaluate_test.rc"))" > "$HERE/planted/verdict.txt"
else
  echo "planted file was replaced or removed (rc=$(cat "$HERE/planted/evaluate_test.rc"))" > "$HERE/planted/verdict.txt"
fi
cat "$HERE/planted/verdict.txt"
rt_reset
