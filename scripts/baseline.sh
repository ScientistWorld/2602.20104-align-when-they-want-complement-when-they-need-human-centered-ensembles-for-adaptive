#!/bin/bash
# Run the paper's single-model baselines (standard AI, behaviour-aware single
# AI, aligned AI, complementary AI; logistic regression) and write their
# predictions to tmp/baselines/<baseline>/<experiment>/predictions.csv.
#
# The scores the gym compares against come from copies of these predictions
# frozen in eval/train/baselines/ and eval/test/baselines/ (written by
# scripts/build_world.sh); predictions an agent writes under a baseline name
# in checkpoints/ are ignored by the evaluators.
#
# Usage:
#   bash scripts/baseline.sh

set -e

cd "$(dirname "$0")/.."

python method/adaptive_team_ai.py --data-root data --baselines-only --out-root tmp/baselines
