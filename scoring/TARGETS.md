# Evaluation Targets

## Primary: college_admission_core

Tests final human-AI team performance in the synthetic two-region admission
task at the paper's alpha=0.9 imperfect-human setting.

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

## Expansion: college_admission_alpha_0_7

Tests the same team-decision objective when humans are less accurate in the
aligned region. This keeps the benchmark in the same problem family while
checking that adaptive assistance still improves the team under a second human
competence regime.

- **Metric**: `team_accuracy` (higher is better)
- **Constraint metrics**: `reliance_probability`, `routing_accuracy`
- **Run**: `bash scripts/reproduce.sh`
