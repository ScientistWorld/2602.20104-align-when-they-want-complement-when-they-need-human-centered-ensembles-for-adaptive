#!/bin/bash
# Shortcut: edit data/ (operating point and ground truth); reference predictions.
set -eu
source /home/user/redteam/common/lib.sh
rt_snapshot
HERE=$RT/shortcut_edit_test_data

for mode in threshold labels_and_humans; do
  rt_reset
  python "$HERE/make.py" "$mode"
  rt_eval "$HERE/$mode"
done
rt_reset
