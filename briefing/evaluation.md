# Evaluation

Evaluation reads held-out predictions written by a method run and computes:

- `team_accuracy`: accuracy of final CGPR team decisions.
- `reliance_probability`: one minus the AI-human disagreement rate in the
  aligned region, the paper's operational trust/reliance quantity.
- `routing_accuracy`: for adaptive systems, how often the selected specialist
  matches the known synthetic region.

`scripts/evaluate.sh` is method-agnostic: it imports only `eval/` code and reads
`checkpoints/college_admission/predictions.csv` by default. Future methods can
be scored by writing the same prediction columns and running the same evaluator.
