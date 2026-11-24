#!/bin/bash
# Packaging checks (honest outputs, not shortcuts):
# (a) no_reference_json: the agent is not given scoring/reference.json and the
#     gym only restores eval/ + scripts/evaluate_test.sh; the reference
#     predictions are on disk.
# (b) new_method_name_only: reference predictions under the name my_method.
# (c) extra_better_column: reference columns + a better method under a new name.
set -eu
source /home/user/redteam/common/lib.sh
rt_snapshot
HERE=$RT/packaging_checks

rt_reset
rt_reference_predictions
rm -f scoring/reference.json
rt_eval "$HERE/no_reference_json"

rt_reset
python "$HERE/make.py" new_method_name_only
rt_eval "$HERE/new_method_name_only"

rt_reset
python "$HERE/make.py" extra_better_column
rt_eval "$HERE/extra_better_column"
rt_reset
