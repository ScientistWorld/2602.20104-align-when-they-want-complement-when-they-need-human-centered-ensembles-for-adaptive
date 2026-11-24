# Red-team audit: does this gym reward better research?

**Short answer: no.**
- The reference already sits almost at the ceiling of what any honest method can reach. The whole honest headroom is +0.012 raw rank score.
- Genuinely better variants (more data) gain +0.001 to +0.003. That is inside the 3-seed spread (sd 0.0047), because the α=0.5 experiment alone moves by ±0.009 between training seeds.
- Weaker variants do score clearly lower.
- Many shortcuts that need no better method score between +0.10 and +0.32. That is 10–30× the honest headroom.
- The hidden test is not hidden. Every test answer is in the agent-visible `data/`, `data/` is never restored, and the scorer trusts whatever rows, columns and settings it finds there.

All numbers below come from one CPU job (`bash redteam/run_all.sh`).

**How each run works:**
1. Reset the workspace to what the agent receives.
2. Apply the variant or shortcut, as the agent would.
3. Put back the gym's `eval/` and `scripts/evaluate_test.sh`.
4. Run `bash scripts/evaluate_test.sh` with no arguments.

Each run keeps its scores in `redteam/<run>/<sub>/scores_test.json`. `rank.json` in the same folder compares those scores with the gym's `scoring/scores_test.json`, using `redteam/common/rank.py`. The full table is in `redteam/summary.md`.

**How the score is computed.** score = Σ_experiments weight × Σ_metrics |coef| × (run − reference), over `team_accuracy` (1.0), `reliance_probability` (0.4) and `routing_accuracy` (0.2). It is raw: the per-metric division by the spread across future attempts is unknown here.
- The headline column is the main proposed method, `adaptive_oracle` (**AO**).
- "mean(5)" averages over all five method names the test scores, baselines included.

## 1. Do better results score higher?

| variant | what changes | score AO | score AO excluding α=0.5 | verdict |
|---|---|---|---|---|
| data_scaling/x10_C1000 | 10× training rows (gym generator) + weaker L2 | +0.0014 | +0.0033 | better, but **inside seed spread** overall |
| data_scaling/x10 | 10× training rows | +0.0007 | +0.0026 | better, but **inside seed spread** overall |
| data_scaling/x3 | 3× training rows | +0.0034 | +0.0025 | better, but **inside seed spread** overall |
| data_scaling/x1 | replica of the reference | 0.0000 | 0.0000 | identical to reference (sanity check) |
| data_scaling/x0.2 | 20% of training rows | −0.0135 | −0.0050 | lower |
| data_scaling/x0.05 | 5% of training rows | −0.0199 | −0.0066 | lower; on **core** it scores *above* the reference (+0.0007) |
| data_scaling/x0.01 | 1% of training rows (~70) | −0.0509 | −0.0455 | clearly lower |
| weaker_no_routing/route_copied_from_region | key idea removed (single model) + route column copied from test `region` | −0.0826 | −0.0775 | clearly lower |
| weaker_no_routing/no_route_column | key idea removed, no route column | −0.2826 | −0.2475 | clearly lower (−0.2 of it is the free routing metric) |
| weaker_rrs/rrs | paper's Rational Routing Shortcut instead of oracle region | −0.1913 | −0.1716 | clearly lower |
| **ceiling**: shortcut_generator_rule | exact label rule from `data/college_admission.py` (perfect AI, accuracy 1.000) | **+0.0120** | +0.0070 | the most any method that sees only features + region can gain |

**Reading the table:**
- Removing training data or removing the paper's idea lowers the score well beyond the noise. The score does point the right way for weaker methods.
- Improvements cannot be told apart from noise. Over all five experiments, every better variant is inside the bootstrap spread (sd 0.0047, range 0.0105).
- Leaving out the α=0.5 experiment, the better variants do clear the spread (+0.0025 to +0.0033 against sd ≈ 0.0003). But the absolute gains are about 0.1–0.2 accuracy points of team accuracy on core.
- The score barely reacts to a 20× cut in training data on core: x0.05 scores +0.0007 there. The two-feature linear problem is solved by logistic regression from a few hundred rows.

**Why there is no headroom:**
- **AI accuracy.** The reference's routed specialists are already 98.3–99.7% accurate on the hidden test. The exception is α=0.5, at 91%, where the aligned specialist is fit to pure-noise targets.
- **Reliance.** Human errors are i.i.d., so a method that sees only the features cannot agree with the human more often than the human is correct. The reference is within 0.003–0.018 of that bound for α ≥ 0.7. At α=0.5 the gap is 0.027, but reliance there is chance anyway.
- **Routing.** The reference copies the test file's `region` column, so routing is exactly at its bound of 1.0.

**Metrics already at their bound (reference):**
- `routing_accuracy` = 1.0 in every experiment. It is obtained by copying an input column.
- `reliance_probability` is within 0.003–0.018 of the features-only bound for α ≥ 0.7: core 0.873 vs 0.891, α=1.0 0.997 vs 1.000. At α=0.5 it is 0.514 vs 0.541, and reliance there is chance.
- `team_accuracy` is within 0.0014–0.023 of the features-plus-region ceiling: core 0.794 vs 0.798, α=0.5 0.612 vs 0.635, α=1.0 0.879 vs 0.880.

**Baselines.** The simple baselines do sit clearly below the reference (single model −0.04 team on core). The scale has no room above it, though, except through shortcuts.

**The α=0.5 experiment is noise.** In the α=0.5 regime humans are at chance, so the aligned specialist is trained on pure-noise targets. Its region-1 boundary is arbitrary and changes with the training sample:
- AO team accuracy at α=0.5 ranges over 0.055 across three bootstrap seeds.
- The AO-vs-single gap there ranges from −0.012 to +0.043.
- This one experiment (weight 0.15) dominates the run-to-run variance of the total score.

## 2. Seed spread and reproduction check

**Seed spread (`redteam/seed_spread`):**
- **Method seed.** The method's `--seed` 42/43/44 gives identical test scores (spread 0.0000), because lbfgs logistic regression is deterministic. The method seed is therefore not a source of variance.
- **Bootstrap seeds.** Resampling the training set with seeds 1/2/3 gives a total AO score of mean −0.0021, sd 0.0047, range 0.0105. Per experiment:

| experiment | team (sd) | reliance (sd) |
|---|---|---|
| core | 0.7942 (0.0003) | 0.8700 (0.0024) |
| α=0.5 | 0.6017 (**0.0244**) | 0.5069 (0.0170) |
| α=0.7 | 0.6897 (0.0012) | 0.6820 (0.0043) |
| α=0.8 | 0.7520 (0.0005) | 0.7978 (0.0022) |
| α=1.0 | 0.8788 (0.0000) | 0.9956 (0.0006) |

- **Evaluation-draw noise.** The CGPR accept/ignore coin flips come from one fixed `team_seed=0`. Across seeds 0–499, the reference's core team accuracy has sd 0.0027 and spans 0.793–0.809. Seed 0 (0.794) happens to sit below the mean (0.801). This noise is larger than the honest headroom on core (+0.004 team).

**Reproduction (`redteam/reference_rerun`):**
- A fresh run of the gym's method on the tracked data matches `scoring/scores_test.json` exactly.
- The shipped checkpoints also match exactly.
- Regenerating the data with `reproduce.sh`'s protocol gives byte-identical files for α ∈ {0.5, 0.7, 0.8, 0.9}.

**Does the method beat its baseline in the paper's direction?** Yes, with caveats.
- Adaptive (oracle) minus single-model team accuracy on the hidden test is +0.032, +0.013, +0.040, +0.042, +0.063 for α = 0.5, 0.7, 0.8, 0.9, 1.0.
- The authors' released numbers are +0.025, +0.022, +0.032, +0.040, +0.061.
- So the gain grows with α as the paper claims, except for the α=0.7 dip and the noisy α=0.5 point (−0.012 to +0.043 over bootstrap seeds).

**Departures from the paper's method and settings:**
1. **The scored "proposed" method is oracle routing on the synthetic group label read from the test file.** This is the paper's default for this synthetic experiment ("group membership is treated as known"). The paper's practical router, RRS, is computed by `method/` (`adaptive_rrs`) but is never scored.
2. **RRS has no access to the region at all.** Region is assigned independently of the features, so RRS routing accuracy is 0.48–0.49 (chance). RRS gains only +0.004 to +0.006 team over the single model on core, and gives up 0.14–0.17 reliance compared with oracle routing. The paper's "RRS ≈ oracle" claim is not testable in this data.
3. **The weighted specialists are fit on all training rows** with weights C^h and 1−C^h (as in the authors' released code). The paper's text fits them on D_a and D_c respectively. The unweighted `adaptive_oracle` specialists are the authors' `team_adaptive_perfect`.
4. **Reliance and routing use the synthetic group (`region == 1`).** The paper defines D_a = {C^h > τ}, and its oracle routes on C^h > τ. At α=0.9, half of group 0 also has C^h > τ.
5. **One test split and one CGPR draw (team_seed 0) per experiment.** The hidden test is the odd-hash half of the same `test.csv` (≈1,500 rows).
6. **The α=1.0 experiment reads an older draw.** `data/college_admission_alpha_1_0` was generated by a different protocol: different rows, no `team_seed`/`sweep_protocol` columns. `reproduce.sh` regenerates `data/college_admission_alpha_1` instead, which no experiment reads (`redteam/reference_rerun/regen_compare.txt`).

**Synthetic data:** it is the paper's own generator and setting, and it does show the paper's effect (adaptive > single). But the problem is saturated:
- region is given at test time;
- the boundary is a linear rule in two features;
- the AI recommendation is consulted only on low-confidence humans.

A better method on the paper's real problem (e.g. inferring region from features, as RRS tries to) cannot show up here. Region is not inferable from the features, and the reference already reads it from the test file.

**Brute force.** The paper's method exists to avoid needing the region / human confidence. The brute force is to use the region and the generator's rule directly (`shortcut_generator_rule`). It solves the test: AI accuracy 1.000, +0.012 over the reference, which is the honest ceiling. The test cannot separate a learned method from a hard-coded one.

## 3. Rank mechanics

- **A failed run writes nothing, and a stale file survives.** If any experiment's predictions file or required column is missing, `eval/test/evaluate.py` raises before writing. `scoring/scores_test.json` is written only at the very end and is never cleared, so a file the agent planted survives (`shortcut_planted_scores`: rc=1, planted file unchanged).
- **Missing route column:** `routing_accuracy` = 0 (its worst value, good).
- **NaN or non-class predictions:** no crash. NaN is cast to a huge negative integer and counted as wrong, which gives worst-like values (good).
- **Missing rows are skipped, not penalised.** Rows absent from `predictions.csv` are silently dropped by the inner join.
- **Only five names are written.** The test writes only the five method names listed in `scoring/reference.json`. A new method under any other name is ignored (`packaging_checks/extra_better_column`: a perfect extra method → score 0). Predictions under only a new name crash the whole test (`new_method_name_only`).
- **Agents cannot learn the five names.** The agent is never given `scoring/reference.json`, and no agent-visible document lists the names. A from-scratch agent cannot know them.
- **The test cannot run without `reference.json`.** Without the file the test crashes (`packaging_checks/no_reference_json`). The visible `evaluate_train.sh` has the same dependency. The gym does not say it restores the file, so the agent's own copy (or none) is used.
- **Directions and signs are fine.** All metrics are `higher_is_better: true`, every coefficient is positive, and the diagnostic metrics have coefficient 0.
- **A useless AI can outrank the reference on single experiments.** An AI that copies the human's input everywhere (it adds nothing) loses overall (−0.014), but beats the reference on α=0.5 (+0.015) and α=0.7 (+0.009) through the 0.4-weighted reliance metric.
- **The routing metric is free.** Any method gets 1.0 by copying `region` (`weaker_no_routing/route_copied_from_region`).

## 4. Shortcuts (each is a real run under `redteam/`, scored by the hidden test)

Reference = 0.0000. Honest ceiling = +0.0120. Bootstrap sd = 0.0047.

| shortcut | what it does | score AO | score mean(5) | works? |
|---|---|---|---|---|
| shortcut_subset_rows/with_labels | reference predictions, but `predictions.csv` lists only rows where the human and AI are right (≈45–75% of rows) | **+0.3238** | +0.3033 | **works**: team acc 1.0, reliance 1.0 |
| shortcut_planted_scores | write `scoring/scores_test.json` with all metrics 1.0, drop one required column so the scorer crashes before writing | **+0.3238** | +0.4994 | **works if the rank reads the file despite rc=1**: the planted file survives unchanged |
| shortcut_subset_rows/no_labels | same, choosing rows only from input columns (region-1 rows where AI agrees with human, region-0 low-confidence rows) | **+0.3094** | +0.2444 | **works**, no labels needed |
| shortcut_edit_test_data/labels_and_humans | edit `data/*/test.csv`: label := AI prediction, region-1 human_pred := AI prediction | **+0.2005** | +0.1517 | **works** (`data/` not restored) |
| shortcut_boost_baselines | overwrite the three baseline columns with the reference's adaptive output (the agent writes the baselines) | 0.0000 | **+0.1717** | **works if baseline entries count in the rank**; AO ties (shortcut) |
| shortcut_edit_test_data/threshold | set `threshold` in `data/*/metadata.csv` to 1.0 (every human treated as unsure) | **+0.1491** | +0.0842 | **works**: one number, team +0.12 to +0.17 |
| shortcut_regenerate_easier_humans | regenerate the scored data dirs with the gym's generator at `complementary_accuracy=0.9`, run the unchanged reference method | **+0.1061** | +0.1274 | **works** |
| shortcut_read_test_labels | region 1: human's test-time prediction; region 0: `label` column of `test.csv` | **+0.1034** | +0.0453 | **works** |
| shortcut_echo_human | region 1: the human's own test-time prediction (an input column); region 0: reference specialist; no labels read | **+0.1017** | +0.0406 | **works**: reliance 1.0. The paper's AI sees only x (y_m = m(x)), so echoing y_h is outside the task, yet nothing forbids it and problem.md lists the human prediction as instance data |
| shortcut_team_seed_search | try `team_seed` 0..499 in `data/*/metadata.csv` on the hidden slice (rule and salt are in `eval/train/evaluate.py`), keep the best | **+0.0132** | +0.0129 | **works**: more than the entire honest ceiling (+0.0120) |
| shortcut_generator_rule | hard-code the generator's label rule from (gpa, score, region) | +0.0120 | +0.0088 | **works** (≈2.5 sd; equals the honest ceiling; a learned method can never beat it) |
| shortcut_trim_reference_json/agent_reference_json | method produced only the core experiment; agent writes its own `scoring/reference.json` listing only core | 0.0000 (60 entries missing, skipped) | 0.0000 | **works**: tie instead of a failed run (control with the gym's file: crash, no scores) |
| degenerate_outputs/copy_human_input | AI = human's input prediction everywhere, route = region | −0.0137 | −0.0015 | fails overall; wins on α=0.5 and α=0.7 |
| degenerate_outputs/constant_zero | predict 0, no route | −0.4169 | −0.1628 | fails |
| degenerate_outputs/nan_values | NaN predictions | −0.4137 | −0.1615 | fails (no crash; counted as wrong) |
| degenerate_outputs/impossible_values | class 2, route 7 | −0.6137 | −0.2415 | fails |

**Ties:**
- `packaging_checks/extra_better_column` (0.0000) is an honest *better* method the test cannot see.
- `data_scaling/x3`, `x10` and `x10_C1000` are honest alternatives the test cannot tell apart from the reference.
- `shortcut_trim_reference_json/agent_reference_json` and `shortcut_boost_baselines` (AO column) are shortcuts.

## 5. What to fix next iteration

1. **Take the answers out of `data/`.** Give the agent the hidden half's features only, plus region if oracle routing stays the task. Keep `label`, `human_pred`, `human_conf`, `threshold` and `team_seed` for the hidden half under `eval/test/`, which is restored, and read them only from there. The CGPR simulation settings must not come from agent-controlled `data/*/metadata.csv`. Change the split salt so the visible train rule does not reveal hidden membership.
2. **Require exactly one binary prediction per hidden `example_id`.** A missing, duplicate or non-binary row should give that experiment its worst values, not be dropped. Delete `scoring/scores_test.json` before scoring, and write worst values per failed experiment instead of crashing.
3. **Fix the output contract.** Use one documented contract (e.g. a `prediction` column and an optional `route_aligned` column), document it in `briefing/evaluation.md`, and ship or restore `scoring/reference.json`. Do not rank agent-written baseline columns; freeze the baselines' scores in the gym.
4. **Do not put the human's test-time prediction in the agent's inputs.** The paper's AI sees only x.
5. **Restore headroom.** Either hide the region at test time and make region inference / RRS the task, with a region that is partly predictable from the features so routing measures something real; or make the label rule harder (non-linear, more features, fewer training rows). Score expected team accuracy (reliance × AI accuracy on low-confidence rows) or average many CGPR draws, instead of one coin-flip draw. Drop the α=0.5 experiment, or average it over data seeds; it is noise only.
6. **Regenerate α=1.0 with the same sweep protocol as the other experiments.**

## Reproduce

Run `bash redteam/run_all.sh` (CPU job, ~55 min, most of it workspace copying). The runner snapshots the gym into `tmp/redteam_pristine/` and runs every `redteam/<run>/run.sh`. It then restores the shipped state and writes `redteam/summary.md`. Each `run.sh` can also be run on its own.
