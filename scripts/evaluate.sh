#!/bin/bash
# Standard evaluation entry point (method-agnostic; imports only eval/ code).
#
# Scores every method directory under checkpoints/ on the visible dev split
# and writes scoring/scores.json (same contract and numbers as
# scripts/evaluate_train.sh). Prediction format per method and experiment:
#   checkpoints/<method>/<experiment>/predictions.csv  with columns
#   example_id,prediction   (prediction in {0,1}; one row per dev and
#   test-input example_id). The ranked entry is checkpoints/submission/.

set -e

cd "$(dirname "$0")/.."

mkdir -p scoring
python eval/train/evaluate.py --output scoring/scores.json
