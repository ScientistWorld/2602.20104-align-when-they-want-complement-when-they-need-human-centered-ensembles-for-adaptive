# Hardening checks (hidden test, raw rank gain vs the reference `submission`)

Raw gain = sum_e weight_e * sum_m |coef_m| * (run - reference); positive = ranks above the reference.
Bootstrap: paired resampling of hidden-test instances (300 draws).

| run | category | raw gain | paired SE | P(gain>0) | dev-slice gain | Δteam admission_group_hidden | Δteam admission_group_hidden_low_reliability | Δteam admission_group_known | Δteam admission_group_known_low_reliability | failures |
|---|---|---|---|---|---|---|---|---|---|---|
| generator_bayes_policy | ceiling (hidden generator) | +0.0307 | 0.0007 | 1.00 | +0.0287 | +0.0333 | +0.0363 | +0.0218 | +0.0242 |  |
| hybrid_comp_gbm_router_gbm | better | +0.0134 | 0.0006 | 1.00 | +0.0128 | +0.0146 | +0.0129 | +0.0080 | +0.0076 |  |
| gbm_team | better | +0.0072 | 0.0008 | 1.00 | +0.0084 | +0.0156 | +0.0110 | +0.0068 | +0.0043 |  |
| gbm_specialists | better | +0.0070 | 0.0007 | 1.00 | +0.0073 | +0.0140 | +0.0083 | +0.0061 | +0.0008 |  |
| hybrid_comp_gbm | better | +0.0070 | 0.0004 | 1.00 | +0.0063 | +0.0055 | +0.0084 | +0.0080 | +0.0076 |  |
| proxy_gbm | better | +0.0060 | 0.0005 | 1.00 | +0.0068 | +0.0088 | +0.0035 | +0.0000 | +0.0000 |  |
| ref_x3data | better (more data) | +0.0015 | 0.0003 | 1.00 | +0.0003 | -0.0001 | +0.0027 | +0.0006 | +0.0010 |  |
| gbm_behavior_aware | simple | +0.0013 | 0.0007 | 0.98 | +0.0012 | +0.0037 | +0.0055 | +0.0064 | +0.0046 |  |
| sc_copy_dev_labels | shortcut | +0.0000 | — | — | +0.0252 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |  |
| submission | reference | +0.0000 | — | — | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |  |
| sb_boost_baselines | shortcut (sandbox) | +0.0000 | — | — | — | +0.0000 | +0.0000 | +0.0000 | +0.0000 |  |
| sb_edit_data | shortcut (sandbox) | +0.0000 | — | — | — | +0.0000 | +0.0000 | +0.0000 | +0.0000 |  |
| sb_no_reference_json | shortcut (sandbox) | +0.0000 | — | — | — | +0.0000 | +0.0000 | +0.0000 | +0.0000 |  |
| ref_boot4 | seed spread | -0.0010 | 0.0003 | 0.00 | -0.0020 | -0.0010 | -0.0003 | +0.0002 | -0.0014 |  |
| ref_boot3 | seed spread | -0.0013 | 0.0003 | 0.00 | -0.0021 | -0.0018 | -0.0004 | -0.0010 | +0.0006 |  |
| ref_boot1 | seed spread | -0.0015 | 0.0003 | 0.00 | -0.0037 | +0.0007 | -0.0027 | -0.0017 | -0.0039 |  |
| ref_boot5 | seed spread | -0.0025 | 0.0004 | 0.00 | -0.0025 | -0.0023 | -0.0023 | -0.0017 | -0.0011 |  |
| ref_boot2 | seed spread | -0.0029 | 0.0004 | 0.00 | -0.0027 | -0.0037 | -0.0007 | -0.0016 | -0.0027 |  |
| lr_team | variant (team-aware rule on LR estimates) | -0.0051 | 0.0005 | 0.00 | -0.0039 | +0.0004 | -0.0049 | +0.0000 | +0.0000 |  |
| gbm_standard | simple | -0.0054 | 0.0008 | 0.00 | -0.0043 | -0.0023 | +0.0063 | -0.0001 | +0.0085 |  |
| ref_10pct | weaker (10% data) | -0.0157 | 0.0006 | 0.00 | -0.0165 | -0.0145 | -0.0003 | -0.0095 | -0.0195 |  |
| handrule_paper | hand-written | -0.0430 | 0.0010 | 0.00 | -0.0402 | -0.0451 | -0.0321 | -0.0293 | -0.0177 |  |
| handrule_score_heuristic | hand-written | -0.0536 | 0.0010 | 0.00 | -0.0500 | -0.0554 | -0.0388 | -0.0334 | -0.0193 |  |
| rrs | weaker (paper's RRS router) | -0.0594 | 0.0011 | 0.00 | -0.0649 | -0.0954 | -0.0512 | +0.0000 | +0.0000 |  |
| aligned_everywhere_gbm | simple | -0.1743 | 0.0016 | 0.00 | -0.1772 | -0.1831 | -0.1439 | -0.1938 | -0.1672 |  |
| sc_missing_experiment | shortcut | -0.1954 | — | — | -0.1954 | +0.0000 | -0.7863 | +0.0000 | +0.0000 | admission_group_hidden_low_reliability: ValueError: predictions.csv missing |
| sc_constant_zero | shortcut | -0.2896 | — | — | -0.2959 | -0.1906 | -0.1593 | -0.2168 | -0.1798 |  |
| sc_duplicate_rows | shortcut | -1.0810 | — | — | -1.0860 | -0.8904 | -0.7863 | -0.9174 | -0.8062 | admission_group_hidden: ValueError: duplicate example_id rows; admission_group_hidden_low_reliability: ValueError: duplicate example_id rows; admission_group_known: ValueError: duplicate example_id rows; admission_group_known_low_reliability: ValueError: duplicate example_id rows |
| sc_impossible_value | shortcut | -1.0810 | — | — | -1.0860 | -0.8904 | -0.7863 | -0.9174 | -0.8062 | admission_group_hidden: ValueError: prediction values must be exactly 0 or 1; admission_group_hidden_low_reliability: ValueError: prediction values must be exactly 0 or 1; admission_group_known: ValueError: prediction values must be exactly 0 or 1; admission_group_known_low_reliability: ValueError: prediction values must be exactly 0 or 1 |
| sc_nan_values | shortcut | -1.0810 | — | — | -1.0860 | -0.8904 | -0.7863 | -0.9174 | -0.8062 | admission_group_hidden: ValueError: prediction values must be exactly 0 or 1; admission_group_hidden_low_reliability: ValueError: prediction values must be exactly 0 or 1; admission_group_known: ValueError: prediction values must be exactly 0 or 1; admission_group_known_low_reliability: ValueError: prediction values must be exactly 0 or 1 |
| sc_probabilities | shortcut | -1.0810 | — | — | -1.0860 | -0.8904 | -0.7863 | -0.9174 | -0.8062 | admission_group_hidden: ValueError: prediction values must be exactly 0 or 1; admission_group_hidden_low_reliability: ValueError: prediction values must be exactly 0 or 1; admission_group_known: ValueError: prediction values must be exactly 0 or 1; admission_group_known_low_reliability: ValueError: prediction values must be exactly 0 or 1 |
| sc_subset_rows | shortcut | -1.0810 | — | — | +0.0000 | -0.8904 | -0.7863 | -0.9174 | -0.8062 | admission_group_hidden: ValueError: 10024 scored example_ids missing; admission_group_hidden_low_reliability: ValueError: 9869 scored example_ids missing; admission_group_known: ValueError: 9977 scored example_ids missing; admission_group_known_low_reliability: ValueError: 10072 scored example_ids missing |
| sb_planted_scores_no_submission | shortcut (sandbox) | -1.0810 | — | — | — | -0.8904 | -0.7863 | -0.9174 | -0.8062 |  |
| sb_new_name_only | packaging (sandbox) | — | — | — | — | — | — | — | — |  |

## Sandbox checks

- **sb_planted_scores_no_submission**: {"category": "shortcut (sandbox)", "planted_file_replaced": true, "rc": 0} — raw gain of `submission`: -1.0810
- **sb_boost_baselines**: {"baseline_scores_unchanged": true, "category": "shortcut (sandbox)", "rc": 0} — raw gain of `submission`: +0.0000
- **sb_edit_data**: {"category": "shortcut (sandbox)", "rc": 0} — raw gain of `submission`: +0.0000
- **sb_no_reference_json**: {"category": "shortcut (sandbox)", "experiments_scored": ["admission_group_hidden", "admission_group_hidden_low_reliability", "admission_group_known", "admission_group_known_low_reliability"], "rc": 0} — raw gain of `submission`: +0.0000
- **sb_new_name_only**: {"category": "packaging (sandbox)", "my_method": {"missing": [], "per_experiment": {"admission_group_hidden": {"reliance_probability": 0.0, "team_accuracy": 0.0}, "admission_group_hidden_low_reliability": {"reliance_probability": 0.0, "team_accuracy": 0.0}, "admission_group_known": {"reliance_probability": 0.0, "team_accuracy": 0.0}, "admission_group_known_low_reliability": {"reliance_probability": 0.0, "team_accuracy": 0.0}}, "total": 0.0}, "rc": 0, "submission": {"missing": [], "per_experiment": {"admission_group_hidden": {"reliance_probability": -0.9386365398723842, "team_accuracy": -0.8903869041430509}, "admission_group_hidden_low_reliability": {"reliance_probability": -0.7631285077729126, "team_accuracy": -0.7862591408911102}, "admission_group_known": {"reliance_probability": -0.9814958074050799, "team_accuracy": -0.9173947914703862}, "admission_group_known_low_reliability": {"reliance_probability": -0.796966597411844, "team_accuracy": -0.8061933136531448}}, "total": -1.0809535456874044}} — raw gain of `submission`: —
- baselines identical in the variant sandbox: True

Reference retrained on 5 bootstrap resamples of the training rows: raw gain mean -0.0018, sd 0.0008, range 0.0019.
