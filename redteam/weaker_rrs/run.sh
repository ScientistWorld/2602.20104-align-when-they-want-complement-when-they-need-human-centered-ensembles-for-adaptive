#!/bin/bash
# The paper's practical router: Rational Routing Shortcut (choose the more
# confident specialist) instead of the oracle region.  Same specialists as the
# reference; only the routing changes (no access to the test region column).
set -eu
source /home/user/redteam/common/lib.sh
rt_snapshot
HERE=$RT/weaker_rrs

rt_reset
python "$RT/common/variant.py" --proposed rrs --info "$HERE/rrs/variant.json"
rt_eval "$HERE/rrs"
rt_reset
