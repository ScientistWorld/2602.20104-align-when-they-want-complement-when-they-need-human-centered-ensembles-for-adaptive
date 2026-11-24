#!/bin/bash
# Job script: rebuild the gym, reproduce the reference end to end, score it on
# the hidden test, then run the hardening checks (variants, shortcuts, world
# sensitivity) through the hidden test.

set -e

cd /home/user

mkdir -p tmp
bash scripts/build_world.sh
bash scripts/reproduce.sh
bash scripts/evaluate_test.sh
python redteam/v2/run_checks.py
python redteam/v2/design_sweep.py default,wider_gate,sharp_gate,no_red_flag
python validate.py --compare --post-design || echo "validate.py reported issues (reference.json results are filled in after this job)"
