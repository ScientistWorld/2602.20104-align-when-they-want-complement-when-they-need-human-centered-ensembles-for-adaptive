# Evaluation Targets

## Primary: college_admission_core

Tests final human-AI team performance in the synthetic two-region admission
task.

- **Metric**: `team_accuracy` (higher is better)
- **Run**: `bash scripts/reproduce.sh`, then `bash scripts/evaluate.sh`

## Constraint: reliance

Checks whether an assistant preserves aligned-region agreement, which controls
the human reliance probability in the CGPR model.

- **Metric**: `reliance_probability` (higher is better)
- **Run**: `bash scripts/evaluate.sh`

## Constraint: routing

For adaptive systems, checks whether the route selection matches the known
region in the synthetic benchmark.

- **Metric**: `routing_accuracy` (higher is better)
- **Run**: `bash scripts/evaluate.sh`
