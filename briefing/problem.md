# Problem Definition

AI-assisted binary decision making in which a human decision maker keeps the
final say. For every case the AI gives one recommendation (admit = 1,
reject = 0). The human then decides with the following confidence-gated,
probabilistic reliance behaviour:

- Each decision has its own confidence threshold tau, drawn from U[0.6, 0.8].
- If the human's self-confidence on the case exceeds tau, the human keeps
  their own judgement. These are the cases where the human feels sure.
- Otherwise the human accepts the AI's recommendation with probability r,
  and keeps their own judgement otherwise.
- r is trust. It equals the rate at which the AI's recommendations agree
  with the human's own judgement on the cases where the human felt sure.

So an assistant is rewarded in two ways:
- for being right on cases where humans are unsure;
- for agreeing with humans where they feel sure, because that agreement is
  what makes them take its advice at all.

## Data

The setting is college admission. Applicants are described by `gpa`, `score`
(a standardised test score) and six context features: `school_resources`,
`family_income`, `neighborhood_index`, `prep_hours`, `test_retakes` and
`activities`. Applicants come from two groups. The group affects both what
predicts the outcome and how sure and accurate the human evaluators are.

Each experiment directory `data/<experiment>/` holds:

| file | rows | columns |
|---|---|---|
| `train.csv` | 7,000 | `example_id`, features, [`group`], `label`, `human_pred`, `human_conf` |
| `dev.csv` | 3,000 | same as train (a visible development split) |
| `test_inputs.csv` | 20,000 | `example_id`, features, [`group`] (no answers) |
| `task.json` | | feature list, whether `group` is given, the threshold distribution |

- `human_pred` is the evaluator's own initial judgement.
- `human_conf` is the evaluator's self-reported confidence.
- Both are available for training (think of them as pilot-study data). They
  are never available at test time.
- `group` is given (in all three files) only in the `admission_group_known*`
  experiments. In the `admission_group_hidden*` experiments the AI sees
  applicant features only.

`data/experiments.json` lists the scored experiments. The other
`data/college_admission*` directories are not scored. They hold the paper's
original two-feature generator and are used only by an unscored
reproduction check.

## What to produce

For every experiment, write one binary recommendation per `example_id` of
`dev.csv` and `test_inputs.csv` to:

    checkpoints/<method_name>/<experiment>/predictions.csv   # columns: example_id,prediction

- The ranked entry is `checkpoints/submission/`.
- Any other method directory is scored too, under its own name, for
  comparison.
- Any recommendation strategy is allowed, but it may use only the inputs
  available at test time.
