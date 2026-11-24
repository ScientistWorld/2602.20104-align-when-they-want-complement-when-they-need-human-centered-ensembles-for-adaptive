# Hardening checks (hidden test, raw rank gain vs the reference `submission`)

Raw gain = sum_e weight_e * sum_m |coef_m| * (run - reference); positive = ranks above the reference.
Bootstrap: paired resampling of hidden-test instances (300 draws).

| run | category | raw gain | paired SE | P(gain>0) | dev-slice gain | Δteam admission_group_hidden | Δteam admission_group_hidden_low_reliability | Δteam admission_group_known | Δteam admission_group_known_low_reliability | failures |
|---|---|---|---|---|---|---|---|---|---|---|
| generator_bayes_policy | ceiling (hidden generator) | +0.0176 | 0.0006 | 1.00 | +0.0181 | +0.0184 | +0.0252 | +0.0124 | +0.0153 |  |
| ref_x3data | better (more data) | +0.0024 | 0.0003 | 1.00 | +0.0036 | +0.0007 | +0.0038 | +0.0021 | +0.0014 |  |
| proxy_gbm | better | +0.0022 | 0.0004 | 1.00 | +0.0018 | +0.0030 | -0.0002 | +0.0000 | +0.0000 |  |
| sc_copy_dev_labels | shortcut | +0.0000 | — | — | +0.0140 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |  |
| submission | reference | +0.0000 | — | — | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |  |
| sb_boost_baselines | shortcut (sandbox) | +0.0000 | — | — | — | +0.0000 | +0.0000 | +0.0000 | +0.0000 |  |
| sb_edit_data | shortcut (sandbox) | +0.0000 | — | — | — | +0.0000 | +0.0000 | +0.0000 | +0.0000 |  |
| sb_no_reference_json | shortcut (sandbox) | +0.0000 | — | — | — | +0.0000 | +0.0000 | +0.0000 | +0.0000 |  |
| ref_boot1 | seed spread | -0.0010 | 0.0003 | 0.00 | -0.0013 | +0.0002 | -0.0021 | +0.0007 | -0.0041 |  |
| ref_boot4 | seed spread | -0.0013 | 0.0004 | 0.00 | +0.0002 | -0.0013 | -0.0004 | +0.0001 | -0.0025 |  |
| ref_boot3 | seed spread | -0.0018 | 0.0004 | 0.00 | -0.0014 | -0.0020 | -0.0020 | -0.0020 | +0.0007 |  |
| gbm_team | better | -0.0019 | 0.0006 | 0.00 | -0.0002 | +0.0048 | +0.0025 | -0.0001 | -0.0010 |  |
| ref_boot5 | seed spread | -0.0023 | 0.0003 | 0.00 | -0.0009 | -0.0014 | -0.0025 | -0.0014 | -0.0021 |  |
| gbm_specialists | better | -0.0023 | 0.0006 | 0.00 | -0.0013 | +0.0033 | -0.0013 | +0.0005 | -0.0048 |  |
| ref_boot2 | seed spread | -0.0028 | 0.0004 | 0.00 | -0.0004 | -0.0027 | -0.0022 | -0.0008 | -0.0042 |  |
| lr_team | variant (team-aware rule on LR estimates) | -0.0052 | 0.0005 | 0.00 | -0.0040 | -0.0007 | -0.0069 | +0.0000 | +0.0000 |  |
| gbm_behavior_aware | simple | -0.0062 | 0.0007 | 0.00 | -0.0043 | -0.0067 | +0.0008 | -0.0006 | +0.0006 |  |
| gbm_standard | simple | -0.0133 | 0.0007 | 0.00 | -0.0130 | -0.0113 | +0.0012 | -0.0069 | +0.0024 |  |
| ref_10pct | weaker (10% data) | -0.0164 | 0.0006 | 0.00 | -0.0142 | -0.0138 | -0.0035 | -0.0127 | -0.0210 |  |
| handrule_paper | hand-written | -0.0367 | 0.0009 | 0.00 | -0.0330 | -0.0407 | -0.0267 | -0.0162 | -0.0096 |  |
| handrule_score_heuristic | hand-written | -0.0473 | 0.0010 | 0.00 | -0.0430 | -0.0508 | -0.0329 | -0.0210 | -0.0116 |  |
| rrs | weaker (paper's RRS router) | -0.0615 | 0.0010 | 0.00 | -0.0626 | -0.1004 | -0.0545 | +0.0000 | +0.0000 |  |
| aligned_everywhere_gbm | simple | -0.1818 | 0.0016 | 0.00 | -0.1809 | -0.1883 | -0.1502 | -0.2029 | -0.1709 |  |
| sc_missing_experiment | shortcut | -0.1977 | — | — | -0.1975 | +0.0000 | -0.7969 | +0.0000 | +0.0000 | admission_group_hidden_low_reliability: ValueError: predictions.csv missing |
| sc_constant_zero | shortcut | -0.3018 | — | — | -0.3042 | -0.2070 | -0.1680 | -0.2239 | -0.1876 |  |
| sc_duplicate_rows | shortcut | -1.0940 | — | — | -1.0972 | -0.9057 | -0.7969 | -0.9265 | -0.8160 | admission_group_hidden: ValueError: duplicate example_id rows; admission_group_hidden_low_reliability: ValueError: duplicate example_id rows; admission_group_known: ValueError: duplicate example_id rows; admission_group_known_low_reliability: ValueError: duplicate example_id rows |
| sc_impossible_value | shortcut | -1.0940 | — | — | -1.0972 | -0.9057 | -0.7969 | -0.9265 | -0.8160 | admission_group_hidden: ValueError: prediction values must be exactly 0 or 1; admission_group_hidden_low_reliability: ValueError: prediction values must be exactly 0 or 1; admission_group_known: ValueError: prediction values must be exactly 0 or 1; admission_group_known_low_reliability: ValueError: prediction values must be exactly 0 or 1 |
| sc_nan_values | shortcut | -1.0940 | — | — | -1.0972 | -0.9057 | -0.7969 | -0.9265 | -0.8160 | admission_group_hidden: ValueError: prediction values must be exactly 0 or 1; admission_group_hidden_low_reliability: ValueError: prediction values must be exactly 0 or 1; admission_group_known: ValueError: prediction values must be exactly 0 or 1; admission_group_known_low_reliability: ValueError: prediction values must be exactly 0 or 1 |
| sc_probabilities | shortcut | -1.0940 | — | — | -1.0972 | -0.9057 | -0.7969 | -0.9265 | -0.8160 | admission_group_hidden: ValueError: prediction values must be exactly 0 or 1; admission_group_hidden_low_reliability: ValueError: prediction values must be exactly 0 or 1; admission_group_known: ValueError: prediction values must be exactly 0 or 1; admission_group_known_low_reliability: ValueError: prediction values must be exactly 0 or 1 |
| sc_subset_rows | shortcut | -1.0940 | — | — | +0.0000 | -0.9057 | -0.7969 | -0.9265 | -0.8160 | admission_group_hidden: ValueError: 10024 scored example_ids missing; admission_group_hidden_low_reliability: ValueError: 9869 scored example_ids missing; admission_group_known: ValueError: 9977 scored example_ids missing; admission_group_known_low_reliability: ValueError: 10072 scored example_ids missing |
| sb_planted_scores_no_submission | shortcut (sandbox) | -1.0940 | — | — | — | -0.9057 | -0.7969 | -0.9265 | -0.8160 |  |
| sb_new_name_only | packaging (sandbox) | — | — | — | — | — | — | — | — |  |

## Sandbox checks

- **sb_planted_scores_no_submission**: {"category": "shortcut (sandbox)", "planted_file_replaced": true, "rc": 0} — raw gain of `submission`: -1.0940
- **sb_boost_baselines**: {"baseline_scores_unchanged": true, "category": "shortcut (sandbox)", "rc": 0} — raw gain of `submission`: +0.0000
- **sb_edit_data**: {"category": "shortcut (sandbox)", "rc": 0} — raw gain of `submission`: +0.0000
- **sb_no_reference_json**: {"category": "shortcut (sandbox)", "experiments_scored": ["admission_group_hidden", "admission_group_hidden_low_reliability", "admission_group_known", "admission_group_known_low_reliability"], "rc": 0} — raw gain of `submission`: +0.0000
- **sb_new_name_only**: {"category": "packaging (sandbox)", "my_method": {"missing": [], "per_experiment": {"admission_group_hidden": {"reliance_probability": 0.0, "team_accuracy": 0.0}, "admission_group_hidden_low_reliability": {"reliance_probability": 0.0, "team_accuracy": 0.0}, "admission_group_known": {"reliance_probability": 0.0, "team_accuracy": 0.0}, "admission_group_known_low_reliability": {"reliance_probability": 0.0, "team_accuracy": 0.0}}, "total": 0.0}, "rc": 0, "submission": {"missing": [], "per_experiment": {"admission_group_hidden": {"reliance_probability": -0.9508219107906946, "team_accuracy": -0.9057247484943176}, "admission_group_hidden_low_reliability": {"reliance_probability": -0.7672222656995882, "team_accuracy": -0.7968750475863998}, "admission_group_known": {"reliance_probability": -0.9812005153812252, "team_accuracy": -0.9264536444536934}, "admission_group_known_low_reliability": {"reliance_probability": -0.7987428983062471, "team_accuracy": -0.8159631139052169}}, "total": -1.0940049129905618}} — raw gain of `submission`: —
- baselines identical in the variant sandbox: True

Reference retrained on 5 bootstrap resamples of the training rows: raw gain mean -0.0018, sd 0.0007, range 0.0018.
