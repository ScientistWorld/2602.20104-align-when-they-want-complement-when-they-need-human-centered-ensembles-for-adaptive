#!/bin/bash
# Shortcut: an empty entry avoids a loss.  The method only produced outputs for
# the core experiment (the four alpha-sweep experiments failed / were skipped).
# (a) control: gym's scoring/reference.json in place -> what does the test do?
# (b) the agent writes its own scoring/reference.json (it is never given the
#     gym's, and the gym only restores eval/ + scripts/evaluate_test.sh) that
#     lists only the core experiment.
set -eu
source /home/user/redteam/common/lib.sh
rt_snapshot
HERE=$RT/shortcut_trim_reference_json

write_core_only() {
  mkdir -p checkpoints/college_admission
  cp "$PRISTINE/checkpoints/college_admission/predictions.csv" checkpoints/college_admission/predictions.csv
}

rt_reset
write_core_only
rt_eval "$HERE/control_gym_reference_json"

rt_reset
write_core_only
cat > scoring/reference.json <<'EOF'
{
  "experiments": {
    "college_admission_core": {
      "results": {
        "behavior_aware_ai": {}, "aligned_ai": {}, "complementary_ai": {},
        "adaptive_oracle": {}, "adaptive_weighted_oracle": {}
      }
    }
  }
}
EOF
rt_eval "$HERE/agent_reference_json"
rt_reset
