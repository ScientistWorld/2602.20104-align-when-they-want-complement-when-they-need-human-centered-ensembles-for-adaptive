#!/bin/bash
# Run every red-team run in one CPU job, then put the shipped gym state back.
# Each run's log is redteam/<run>/run.log; its scores are under redteam/<run>/<sub>/.
set -u
cd /home/user
source /home/user/redteam/common/lib.sh
rt_snapshot

RUNS="reference_rerun seed_spread data_scaling weaker_no_routing weaker_rrs packaging_checks
shortcut_read_test_labels shortcut_echo_human shortcut_generator_rule shortcut_subset_rows
shortcut_planted_scores shortcut_edit_test_data shortcut_team_seed_search
shortcut_regenerate_easier_humans shortcut_boost_baselines shortcut_trim_reference_json
degenerate_outputs"

for r in $RUNS; do
  echo "=== $r"
  if bash "$RT/$r/run.sh" > "$RT/$r/run.log" 2>&1; then
    grep '^\[rank\]' "$RT/$r/run.log" || true
  else
    echo "RUN FAILED: $r (see redteam/$r/run.log)"
    tail -5 "$RT/$r/run.log"
  fi
done

rt_finish
python "$RT/summarize.py"
