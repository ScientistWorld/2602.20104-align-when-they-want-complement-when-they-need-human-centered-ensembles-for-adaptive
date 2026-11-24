#!/bin/bash
# Key idea removed: no specialists, no routing.  The single behavior-aware
# model is written into both proposed columns.
# (a) no route column (routing_accuracy falls to 0 by the evaluator's rule)
# (b) same predictions, route column copied from the test file's region column
set -eu
source /home/user/redteam/common/lib.sh
rt_snapshot
HERE=$RT/weaker_no_routing

rt_reset
python "$RT/common/variant.py" --proposed single --info "$HERE/no_route_column/variant.json"
rt_eval "$HERE/no_route_column"

rt_reset
python "$RT/common/variant.py" --proposed single_route_region --info "$HERE/route_copied_from_region/variant.json"
rt_eval "$HERE/route_copied_from_region"
rt_reset
