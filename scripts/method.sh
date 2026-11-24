#!/bin/bash
# Run the paper's adaptive ensemble on every experiment and write its
# recommendations to checkpoints/submission/<experiment>/predictions.csv
# (the ranked slot). Deterministic (logistic regression, lbfgs).
#
# Usage:
#   bash scripts/method.sh

set -e

cd "$(dirname "$0")/.."

python method/adaptive_team_ai.py --data-root data --out-root checkpoints --name submission
