#!/bin/bash
# Visible dev-slice entry point. Takes no arguments.
#
# Same measurement as the hidden test, on the visible dev split
# (data/<experiment>/dev.csv). Scores every method directory under
# checkpoints/ plus the ranked slot `submission` and the frozen baselines in
# eval/train/baselines/, and writes scoring/scores_train.json.

set -e

cd "$(dirname "$0")/.."

python eval/train/evaluate.py
