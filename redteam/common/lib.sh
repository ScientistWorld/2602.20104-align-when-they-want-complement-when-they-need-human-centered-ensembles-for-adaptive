#!/bin/bash
# Shared steps for every red-team run.  Source this file from a run.sh.
#
# A run = (1) put the workspace back into the state the gym hands a future agent
# (data/, method/, scoring/ as shipped, no checkpoints, no score files),
# (2) do what the agent would do (write predictions / edit files),
# (3) put the gym's own eval/ and scripts/evaluate_test.sh back, exactly as the
# gym does when the agent signals done, (4) run `bash scripts/evaluate_test.sh`
# with no arguments and keep the scores it writes, (5) score them against the
# reference test scores with redteam/common/rank.py.

WS=/home/user
RT=$WS/redteam
PRISTINE=$WS/tmp/redteam_pristine
EXP_DIRS="college_admission college_admission_alpha_0_5 college_admission_alpha_0_7 college_admission_alpha_0_8 college_admission_alpha_1_0"

# Save the shipped gym state once per job, plus the reference method's outputs.
rt_snapshot() {
  if [ -f "$PRISTINE/.done" ]; then
    return 0
  fi
  rm -rf "$PRISTINE"
  mkdir -p "$PRISTINE/checkpoints" "$PRISTINE/checkpoints_existing" "$PRISTINE/scripts"
  cp -a "$WS/data" "$PRISTINE/data"
  cp -a "$WS/scoring" "$PRISTINE/scoring"
  cp -a "$WS/eval" "$PRISTINE/eval"
  cp -a "$WS/method" "$PRISTINE/method"
  cp -a "$WS/scripts/evaluate_test.sh" "$WS/scripts/evaluate_train.sh" "$WS/scripts/method.sh" "$PRISTINE/scripts/"
  for d in $EXP_DIRS; do
    if [ -d "$WS/checkpoints/$d" ]; then
      cp -a "$WS/checkpoints/$d" "$PRISTINE/checkpoints_existing/$d"
    fi
    # Reference outputs, regenerated with the gym's own (deterministic) method code.
    python "$WS/method/college_admission_method.py" \
      --data-dir "$WS/data/$d" --output-dir "$PRISTINE/checkpoints/$d" --seed 42
  done
  touch "$PRISTINE/.done"
}

# Workspace as the agent receives it (+ the gym's reference.json, which the
# hidden test needs to run at all).
rt_reset() {
  cd "$WS"
  rm -rf "$WS/data" && cp -a "$PRISTINE/data" "$WS/data"
  rm -rf "$WS/method" && cp -a "$PRISTINE/method" "$WS/method"
  rm -rf "$WS/eval" && cp -a "$PRISTINE/eval" "$WS/eval"
  rm -rf "$WS/scoring" && cp -a "$PRISTINE/scoring" "$WS/scoring"
  rm -f "$WS/scoring/scores_test.json" "$WS/scoring/scores_train.json" "$WS/scoring/scores.json"
  cp "$PRISTINE/scripts/evaluate_test.sh" "$PRISTINE/scripts/evaluate_train.sh" "$PRISTINE/scripts/method.sh" "$WS/scripts/"
  for d in $EXP_DIRS; do
    rm -rf "$WS/checkpoints/$d"
  done
  mkdir -p "$WS/checkpoints"
  rm -rf "$WS/tmp/rt_work"
  mkdir -p "$WS/tmp/rt_work"
}

# Copy the reference method's predictions into checkpoints/ (reuse of the
# reference's trained outputs for runs that only change what is written).
rt_reference_predictions() {
  for d in $EXP_DIRS; do
    mkdir -p "$WS/checkpoints/$d"
    cp "$PRISTINE/checkpoints/$d/predictions.csv" "$WS/checkpoints/$d/predictions.csv"
  done
}

# Gym restores eval/ + scripts/evaluate_test.sh, then runs the hidden test.
rt_eval() {
  local outdir="$1"
  mkdir -p "$outdir"
  cd "$WS"
  rm -rf "$WS/eval" && cp -a "$PRISTINE/eval" "$WS/eval"
  cp "$PRISTINE/scripts/evaluate_test.sh" "$WS/scripts/evaluate_test.sh"
  set +e
  (cd "$WS" && bash scripts/evaluate_test.sh) > "$outdir/evaluate_test.log" 2>&1
  local rc=$?
  set -e
  echo "$rc" > "$outdir/evaluate_test.rc"
  rm -f "$outdir/scores_test.json"
  if [ -f "$WS/scoring/scores_test.json" ]; then
    cp "$WS/scoring/scores_test.json" "$outdir/scores_test.json"
  fi
  python "$RT/common/rank.py" --run-dir "$outdir" --rc "$rc" || echo "rank.py failed for $outdir"
}

# End of job: shipped state back, including the gym's own checkpoints and scores.
rt_finish() {
  rt_reset
  cp -a "$PRISTINE/scoring/." "$WS/scoring/"
  for d in $EXP_DIRS; do
    if [ -d "$PRISTINE/checkpoints_existing/$d" ]; then
      rm -rf "$WS/checkpoints/$d"
      cp -a "$PRISTINE/checkpoints_existing/$d" "$WS/checkpoints/$d"
    fi
  done
}
