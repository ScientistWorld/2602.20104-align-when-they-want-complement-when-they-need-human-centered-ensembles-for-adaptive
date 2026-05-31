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
region in the synthetic benchmark. Oracle routing should match by construction
and is the paper-reported synthetic routing condition.

- **Metric**: `routing_accuracy` (higher is better)
- **Run**: `bash scripts/evaluate.sh`

## Expansion: human-reliability sweep

Tests the same team-decision objective across multiple aligned-region human
accuracy levels from the authors' released synthetic alpha sweep. This checks
the paper's secondary claim that adaptive gains increase as the aligned-region
human signal becomes more reliable.

- **Metric**: `team_accuracy` (higher is better)
- **Constraint metrics**: `reliance_probability`, `routing_accuracy`
- **Experiments**: `college_admission_alpha_0_5`, `college_admission_alpha_0_7`, `college_admission_alpha_0_8`, `college_admission_alpha_1_0`
- **Run**: `bash scripts/reproduce.sh`
