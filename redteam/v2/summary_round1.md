# Hardening checks (hidden test, raw rank gain vs the reference `submission`)

Raw gain = sum_e weight_e * sum_m |coef_m| * (run - reference); positive = ranks above the reference.
Bootstrap: paired resampling of hidden-test instances (300 draws).

| run | category | raw gain | paired SE | P(gain>0) | dev-slice gain | Δteam admission_group_hidden | Δteam admission_group_hidden_low_reliability | Δteam admission_group_known | Δteam admission_group_known_low_reliability | failures |
|---|---|---|---|---|---|---|---|---|---|---|
| generator_bayes_policy | ceiling (hidden generator) | +0.0474 | 0.0009 | 1.00 | +0.0489 | +0.0344 | +0.0430 | +0.0321 | +0.0334 |  |
| gbm_team | better | +0.0299 | 0.0009 | 1.00 | +0.0319 | +0.0184 | +0.0195 | +0.0215 | +0.0180 |  |
| gbm_specialists | better | +0.0267 | 0.0009 | 1.00 | +0.0293 | +0.0163 | +0.0179 | +0.0187 | +0.0054 |  |
| gbm_behavior_aware | simple | +0.0264 | 0.0008 | 1.00 | +0.0266 | +0.0096 | +0.0164 | +0.0195 | +0.0189 |  |
| gbm_standard | simple | +0.0121 | 0.0008 | 1.00 | +0.0133 | +0.0030 | +0.0119 | +0.0107 | +0.0142 |  |
| proxy_gbm | better | +0.0024 | 0.0005 | 1.00 | -0.0002 | +0.0023 | -0.0027 | +0.0000 | +0.0000 |  |
| ref_x3data | better (more data) | +0.0010 | 0.0004 | 1.00 | +0.0009 | +0.0005 | +0.0019 | +0.0011 | +0.0009 |  |
| sc_copy_dev_labels | shortcut | +0.0000 | — | — | +0.0441 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |  |
| submission | reference | +0.0000 | — | — | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |  |
| sb_boost_baselines | shortcut (sandbox) | +0.0000 | — | — | — | +0.0000 | +0.0000 | +0.0000 | +0.0000 |  |
| sb_edit_data | shortcut (sandbox) | +0.0000 | — | — | — | +0.0000 | +0.0000 | +0.0000 | +0.0000 |  |
| sb_no_reference_json | shortcut (sandbox) | +0.0000 | — | — | — | +0.0000 | +0.0000 | +0.0000 | +0.0000 |  |
| ref_boot5 | seed spread | -0.0013 | 0.0004 | 0.00 | -0.0008 | -0.0020 | -0.0002 | +0.0002 | -0.0018 |  |
| ref_boot2 | seed spread | -0.0015 | 0.0004 | 0.00 | -0.0021 | -0.0004 | +0.0005 | -0.0007 | -0.0063 |  |
| ref_boot3 | seed spread | -0.0016 | 0.0004 | 0.00 | -0.0018 | -0.0020 | -0.0028 | -0.0009 | +0.0012 |  |
| ref_boot1 | seed spread | -0.0017 | 0.0004 | 0.00 | -0.0024 | -0.0023 | +0.0000 | +0.0013 | -0.0038 |  |
| ref_boot4 | seed spread | -0.0017 | 0.0004 | 0.00 | -0.0014 | -0.0013 | -0.0014 | -0.0002 | -0.0043 |  |
| lr_team | better | -0.0059 | 0.0005 | 0.00 | -0.0047 | -0.0024 | -0.0079 | +0.0000 | +0.0000 |  |
| ref_10pct | weaker (10% data) | -0.0119 | 0.0007 | 0.00 | -0.0133 | -0.0085 | -0.0063 | -0.0109 | -0.0134 |  |
| handrule_paper | hand-written | -0.0212 | 0.0010 | 0.00 | -0.0161 | -0.0328 | -0.0217 | -0.0068 | -0.0049 |  |
| handrule_score_heuristic | hand-written | -0.0232 | 0.0010 | 0.00 | -0.0187 | -0.0398 | -0.0268 | -0.0058 | -0.0044 |  |
| rrs | weaker (paper's RRS router) | -0.0542 | 0.0010 | 0.00 | -0.0558 | -0.0912 | -0.0554 | +0.0000 | +0.0000 |  |
| aligned_everywhere_gbm | simple | -0.1549 | 0.0016 | 0.00 | -0.1531 | -0.1727 | -0.1459 | -0.1872 | -0.1625 |  |
| sc_missing_experiment | shortcut | -0.1937 | — | — | -0.1971 | +0.0000 | -0.7857 | +0.0000 | +0.0000 | admission_group_hidden_low_reliability: ValueError: predictions.csv missing |
| sc_constant_zero | shortcut | -0.2694 | — | — | -0.2707 | -0.1965 | -0.1670 | -0.2096 | -0.1869 |  |
| sc_duplicate_rows | shortcut | -1.0678 | — | — | -1.0711 | -0.8927 | -0.7857 | -0.9059 | -0.8049 | admission_group_hidden: ValueError: duplicate example_id rows; admission_group_hidden_low_reliability: ValueError: duplicate example_id rows; admission_group_known: ValueError: duplicate example_id rows; admission_group_known_low_reliability: ValueError: duplicate example_id rows |
| sc_impossible_value | shortcut | -1.0678 | — | — | -1.0711 | -0.8927 | -0.7857 | -0.9059 | -0.8049 | admission_group_hidden: ValueError: prediction values must be exactly 0 or 1; admission_group_hidden_low_reliability: ValueError: prediction values must be exactly 0 or 1; admission_group_known: ValueError: prediction values must be exactly 0 or 1; admission_group_known_low_reliability: ValueError: prediction values must be exactly 0 or 1 |
| sc_nan_values | shortcut | -1.0678 | — | — | -1.0711 | -0.8927 | -0.7857 | -0.9059 | -0.8049 | admission_group_hidden: ValueError: prediction values must be exactly 0 or 1; admission_group_hidden_low_reliability: ValueError: prediction values must be exactly 0 or 1; admission_group_known: ValueError: prediction values must be exactly 0 or 1; admission_group_known_low_reliability: ValueError: prediction values must be exactly 0 or 1 |
| sc_probabilities | shortcut | -1.0678 | — | — | -1.0711 | -0.8927 | -0.7857 | -0.9059 | -0.8049 | admission_group_hidden: ValueError: prediction values must be exactly 0 or 1; admission_group_hidden_low_reliability: ValueError: prediction values must be exactly 0 or 1; admission_group_known: ValueError: prediction values must be exactly 0 or 1; admission_group_known_low_reliability: ValueError: prediction values must be exactly 0 or 1 |
| sc_subset_rows | shortcut | -1.0678 | — | — | +0.0000 | -0.8927 | -0.7857 | -0.9059 | -0.8049 | admission_group_hidden: ValueError: 10024 scored example_ids missing; admission_group_hidden_low_reliability: ValueError: 9869 scored example_ids missing; admission_group_known: ValueError: 9977 scored example_ids missing; admission_group_known_low_reliability: ValueError: 10072 scored example_ids missing |
| sb_planted_scores_no_submission | shortcut (sandbox) | -1.0678 | — | — | — | -0.8927 | -0.7857 | -0.9059 | -0.8049 |  |
| sb_new_name_only | packaging (sandbox) | — | — | — | — | — | — | — | — |  |

## Sandbox checks

- **sb_planted_scores_no_submission**: {"category": "shortcut (sandbox)", "planted_file_replaced": true, "rc": 0} — raw gain of `submission`: -1.0678
- **sb_boost_baselines**: {"baseline_scores_unchanged": true, "category": "shortcut (sandbox)", "rc": 0} — raw gain of `submission`: +0.0000
- **sb_edit_data**: {"category": "shortcut (sandbox)", "rc": 0} — raw gain of `submission`: +0.0000
- **sb_no_reference_json**: {"category": "shortcut (sandbox)", "experiments_scored": ["admission_group_hidden", "admission_group_hidden_low_reliability", "admission_group_known", "admission_group_known_low_reliability"], "rc": 0} — raw gain of `submission`: +0.0000
- **sb_new_name_only**: {"category": "packaging (sandbox)", "my_method": {"missing": [], "per_experiment": {"admission_group_hidden": {"reliance_probability": 0.0, "team_accuracy": 0.0}, "admission_group_hidden_low_reliability": {"reliance_probability": 0.0, "team_accuracy": 0.0}, "admission_group_known": {"reliance_probability": 0.0, "team_accuracy": 0.0}, "admission_group_known_low_reliability": {"reliance_probability": 0.0, "team_accuracy": 0.0}}, "total": 0.0}, "rc": 0, "submission": {"missing": [], "per_experiment": {"admission_group_hidden": {"reliance_probability": -0.8899929353929337, "team_accuracy": -0.8926582839685847}, "admission_group_hidden_low_reliability": {"reliance_probability": -0.7312933588522398, "team_accuracy": -0.7856653646562161}, "admission_group_known": {"reliance_probability": -0.919619227547399, "team_accuracy": -0.9059418104761419}, "admission_group_known_low_reliability": {"reliance_probability": -0.7776518311647314, "team_accuracy": -0.8049105414746208}}, "total": -1.06782887630365}} — raw gain of `submission`: —
- baselines identical in the variant sandbox: True

Reference retrained on 5 bootstrap resamples of the training rows: raw gain mean -0.0016, sd 0.0002, range 0.0004.
