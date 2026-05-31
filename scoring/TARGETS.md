# Evaluation Targets

## Primary: college_admission_core

Tests final human-AI team performance in the synthetic two-region admission
task at the main alpha=0.9 imperfect-human setting.

- **Metric**: `team_accuracy` (higher is better)
- **Run**: `bash scripts/reproduce.sh`, then `bash scripts/evaluate.sh`

## Constraint: reliance

Checks whether an assistant preserves agreement in the human-reliable regime,
which controls the human reliance probability under the provided evaluation
rule.

- **Metric**: `reliance_probability` (higher is better)
- **Run**: `bash scripts/evaluate.sh`

## Constraint: routing

For methods that emit a route or regime choice, checks whether the selection
matches the known region in the synthetic benchmark.

- **Metric**: `routing_accuracy` (higher is better)
- **Run**: `bash scripts/evaluate.sh`

## Expansion: human-reliability sweep

Tests the same team-decision objective across multiple human-reliable-regime
accuracy levels from the synthetic alpha sweep. This checks the secondary claim
that behavior-aware gains increase as the reliable human signal becomes
stronger.

- **Metric**: `team_accuracy` (higher is better)
- **Constraint metrics**: `reliance_probability`, `routing_accuracy`
- **Experiments**: `college_admission_alpha_0_5`, `college_admission_alpha_0_7`, `college_admission_alpha_0_8`, `college_admission_alpha_1_0`
- **Run**: `bash scripts/reproduce.sh`
