#!/bin/bash
# Maintainer-only: (re)build the gym's data and hidden test.
#
# Needs eval/test/ (hidden from future agents). Regenerates, deterministically:
#   data/<experiment>/{train.csv,dev.csv,test_inputs.csv,task.json}, data/experiments.json
#   eval/test/answers/<experiment>.csv           (hidden-test answers)
#   eval/test/baselines/, eval/train/baselines/  (frozen baseline predictions)

set -e

cd "$(dirname "$0")/.."

python eval/test/world.py build --summary eval/test/build_summary.json
python method/adaptive_team_ai.py --data-root data --baselines-only --out-root tmp/baselines
python eval/test/freeze_baselines.py --source tmp/baselines
python eval/train/freeze_baselines.py --source tmp/baselines
