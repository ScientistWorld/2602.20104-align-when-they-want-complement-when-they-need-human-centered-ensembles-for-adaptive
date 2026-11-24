#!/bin/bash
# Hidden-test entry point. Takes no arguments.
#
# Scores every method directory under checkpoints/ (each holding
# <experiment>/predictions.csv with columns example_id,prediction), always
# including the ranked slot `submission` (scored as failed when absent), plus
# the baselines frozen in eval/test/baselines/. Writes scoring/scores_test.json
# (deleted first, so a failed run never leaves an old file behind).

set -e

cd "$(dirname "$0")/.."

python eval/test/evaluate.py
