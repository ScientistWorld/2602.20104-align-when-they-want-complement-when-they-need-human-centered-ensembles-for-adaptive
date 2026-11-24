#!/bin/bash
# Shortcut: choose team_seed in data/*/metadata.csv; reference predictions.
set -eu
source /home/user/redteam/common/lib.sh
rt_snapshot
HERE=$RT/shortcut_team_seed_search

rt_reset
python "$HERE/make.py" "$HERE/best_seed/seed_search.json"
rt_eval "$HERE/best_seed"
rt_reset
