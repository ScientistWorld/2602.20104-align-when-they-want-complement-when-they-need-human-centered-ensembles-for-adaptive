# Evaluation

Evaluation reads held-out predictions written by a method run and computes:

- `team_accuracy`: accuracy of final human-AI team decisions under the provided
  probabilistic reliance rule.
- `reliance_probability`: one minus the AI-human disagreement rate in the
  high-confidence human-reliable regime.
- `routing_accuracy`: for methods that emit a route or regime choice, how often
  that choice matches the known synthetic regime.

`scripts/evaluate.sh` is method-agnostic: it imports only `eval/` code and reads
`checkpoints/college_admission/predictions.csv` by default. Future methods can
be scored by writing the same prediction columns and running the same evaluator.
