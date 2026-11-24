# World-design sweep (team accuracy / reliance on each world's hidden test)

## world `default`

- admission_group_hidden: human acc privileged 0.900, underprivileged 0.498, privileged share 0.501, mean P(alignment) 0.470, slip 0.015
- admission_group_hidden_low_reliability: human acc privileged 0.751, underprivileged 0.504, privileged share 0.501, mean P(alignment) 0.413, slip 0.197
- admission_group_known: human acc privileged 0.893, underprivileged 0.509, privileged share 0.503, mean P(alignment) 0.472, slip 0.015
- admission_group_known_low_reliability: human acc privileged 0.750, underprivileged 0.508, privileged share 0.500, mean P(alignment) 0.411, slip 0.197

| method | raw gain | admission_group_hidden team / rel / ai | admission_group_hidden_low_reliability team / rel / ai | admission_group_known team / rel / ai | admission_group_known_low_reliability team / rel / ai |
|---|---|---|---|---|---|
| generator_bayes_policy | +0.0474 | 0.9271 / 0.945 / 0.943 | 0.8287 / 0.784 / 0.945 | 0.9380 / 0.972 / 0.956 | 0.8383 / 0.803 / 0.960 |
| gbm_team | +0.0299 | 0.9111 / 0.942 / 0.921 | 0.8052 / 0.769 / 0.915 | 0.9274 / 0.978 / 0.941 | 0.8230 / 0.788 / 0.938 |
| gbm_specialists | +0.0267 | 0.9089 / 0.952 / 0.918 | 0.8035 / 0.771 / 0.912 | 0.9246 / 0.983 / 0.934 | 0.8103 / 0.785 / 0.920 |
| gbm_behavior_aware | +0.0264 | 0.9022 / 0.944 / 0.912 | 0.8021 / 0.766 / 0.910 | 0.9254 / 0.979 / 0.937 | 0.8238 / 0.796 / 0.941 |
| gbm_standard | +0.0121 | 0.8956 / 0.905 / 0.929 | 0.7976 / 0.759 / 0.930 | 0.9166 / 0.930 / 0.962 | 0.8192 / 0.773 / 0.964 |
| proxy_gbm | +0.0024 | 0.8949 / 0.902 / 0.913 | 0.7829 / 0.754 / 0.900 | 0.9059 / 0.920 / 0.931 | 0.8049 / 0.778 / 0.933 |
| reference | +0.0000 | 0.8927 / 0.890 / 0.908 | 0.7857 / 0.731 / 0.894 | 0.9059 / 0.920 / 0.931 | 0.8049 / 0.778 / 0.933 |
| lr_team | -0.0058 | 0.8903 / 0.863 / 0.900 | 0.7778 / 0.710 / 0.875 | 0.9059 / 0.920 / 0.931 | 0.8049 / 0.778 / 0.933 |
| handrule_paper | -0.0212 | 0.8599 / 0.864 / 0.875 | 0.7639 / 0.732 / 0.873 | 0.8991 / 0.918 / 0.938 | 0.8000 / 0.770 / 0.938 |
| handrule_score_heuristic | -0.0232 | 0.8529 / 0.870 / 0.865 | 0.7588 / 0.735 / 0.866 | 0.9001 / 0.926 / 0.936 | 0.8005 / 0.771 / 0.938 |
| rrs | -0.0542 | 0.8014 / 0.803 / 0.781 | 0.7303 / 0.660 / 0.787 | — | — |
| behavior_aware_ai | -0.1077 | 0.8074 / 0.782 / 0.782 | 0.7311 / 0.657 / 0.788 | 0.8100 / 0.774 / 0.785 | 0.7326 / 0.652 / 0.784 |
| standard_ai | -0.1116 | 0.8063 / 0.779 / 0.786 | 0.7185 / 0.689 / 0.786 | 0.8063 / 0.767 / 0.785 | 0.7178 / 0.692 / 0.784 |
| complementary_ai | -0.1223 | 0.8338 / 0.597 / 0.782 | 0.7479 / 0.579 / 0.785 | 0.8320 / 0.592 / 0.776 | 0.7484 / 0.566 / 0.777 |
| aligned_ai | -0.1515 | 0.7341 / 0.921 / 0.742 | 0.6500 / 0.772 / 0.737 | 0.7345 / 0.920 / 0.744 | 0.6509 / 0.778 / 0.735 |
| aligned_everywhere_gbm | -0.1549 | 0.7200 / 0.968 / 0.726 | 0.6398 / 0.776 / 0.707 | 0.7188 / 0.983 / 0.725 | 0.6424 / 0.796 / 0.713 |

## world `wider_gate`

- admission_group_hidden: human acc privileged 0.900, underprivileged 0.500, privileged share 0.499, mean P(alignment) 0.468, slip 0.014
- admission_group_hidden_low_reliability: human acc privileged 0.752, underprivileged 0.503, privileged share 0.495, mean P(alignment) 0.408, slip 0.196
- admission_group_known: human acc privileged 0.893, underprivileged 0.508, privileged share 0.498, mean P(alignment) 0.468, slip 0.014
- admission_group_known_low_reliability: human acc privileged 0.750, underprivileged 0.509, privileged share 0.499, mean P(alignment) 0.411, slip 0.196

| method | raw gain | admission_group_hidden team / rel / ai | admission_group_hidden_low_reliability team / rel / ai | admission_group_known team / rel / ai | admission_group_known_low_reliability team / rel / ai |
|---|---|---|---|---|---|
| generator_bayes_policy | +0.0511 | 0.9272 / 0.948 / 0.943 | 0.8295 / 0.786 / 0.945 | 0.9389 / 0.973 / 0.957 | 0.8385 / 0.803 / 0.960 |
| gbm_team | +0.0333 | 0.9105 / 0.940 / 0.920 | 0.8045 / 0.772 / 0.912 | 0.9290 / 0.978 / 0.942 | 0.8248 / 0.791 / 0.940 |
| gbm_specialists | +0.0295 | 0.9076 / 0.951 / 0.916 | 0.8034 / 0.775 / 0.910 | 0.9253 / 0.984 / 0.934 | 0.8108 / 0.787 / 0.921 |
| gbm_behavior_aware | +0.0281 | 0.8984 / 0.941 / 0.906 | 0.8011 / 0.769 / 0.906 | 0.9276 / 0.980 / 0.939 | 0.8236 / 0.795 / 0.941 |
| gbm_standard | +0.0129 | 0.8910 / 0.898 / 0.922 | 0.7957 / 0.760 / 0.925 | 0.9182 / 0.932 / 0.963 | 0.8191 / 0.774 / 0.963 |
| proxy_gbm | +0.0058 | 0.8939 / 0.902 / 0.912 | 0.7850 / 0.757 / 0.901 | 0.9077 / 0.921 / 0.933 | 0.8041 / 0.777 / 0.931 |
| reference | +0.0000 | 0.8868 / 0.881 / 0.899 | 0.7844 / 0.731 / 0.890 | 0.9077 / 0.921 / 0.933 | 0.8041 / 0.777 / 0.931 |
| lr_team | -0.0060 | 0.8849 / 0.850 / 0.892 | 0.7768 / 0.707 / 0.872 | 0.9077 / 0.921 / 0.933 | 0.8041 / 0.777 / 0.931 |
| handrule_paper | -0.0194 | 0.8569 / 0.860 / 0.870 | 0.7633 / 0.734 / 0.871 | 0.8997 / 0.919 / 0.937 | 0.8008 / 0.771 / 0.938 |
| handrule_score_heuristic | -0.0217 | 0.8492 / 0.865 / 0.860 | 0.7576 / 0.736 / 0.863 | 0.9010 / 0.927 / 0.936 | 0.8012 / 0.772 / 0.938 |
| rrs | -0.0512 | 0.8011 / 0.801 / 0.781 | 0.7307 / 0.661 / 0.788 | — | — |
| behavior_aware_ai | -0.1050 | 0.8077 / 0.778 / 0.782 | 0.7316 / 0.657 / 0.788 | 0.8094 / 0.773 / 0.784 | 0.7335 / 0.653 / 0.785 |
| standard_ai | -0.1087 | 0.8064 / 0.776 / 0.785 | 0.7194 / 0.689 / 0.787 | 0.8065 / 0.767 / 0.785 | 0.7190 / 0.690 / 0.785 |
| complementary_ai | -0.1194 | 0.8335 / 0.596 / 0.783 | 0.7485 / 0.579 / 0.786 | 0.8319 / 0.591 / 0.778 | 0.7489 / 0.567 / 0.777 |
| aligned_ai | -0.1499 | 0.7327 / 0.921 / 0.740 | 0.6482 / 0.774 / 0.733 | 0.7342 / 0.921 / 0.744 | 0.6486 / 0.777 / 0.731 |
| aligned_everywhere_gbm | -0.1523 | 0.7183 / 0.969 / 0.725 | 0.6359 / 0.780 / 0.702 | 0.7201 / 0.984 / 0.726 | 0.6451 / 0.795 / 0.716 |

## world `sharp_gate`

- admission_group_hidden: human acc privileged 0.900, underprivileged 0.498, privileged share 0.503, mean P(alignment) 0.472, slip 0.014
- admission_group_hidden_low_reliability: human acc privileged 0.751, underprivileged 0.504, privileged share 0.501, mean P(alignment) 0.413, slip 0.196
- admission_group_known: human acc privileged 0.893, underprivileged 0.509, privileged share 0.502, mean P(alignment) 0.471, slip 0.014
- admission_group_known_low_reliability: human acc privileged 0.749, underprivileged 0.508, privileged share 0.501, mean P(alignment) 0.411, slip 0.196

| method | raw gain | admission_group_hidden team / rel / ai | admission_group_hidden_low_reliability team / rel / ai | admission_group_known team / rel / ai | admission_group_known_low_reliability team / rel / ai |
|---|---|---|---|---|---|
| generator_bayes_policy | +0.0478 | 0.9298 / 0.950 / 0.948 | 0.8315 / 0.789 / 0.950 | 0.9381 / 0.972 / 0.956 | 0.8386 / 0.803 / 0.960 |
| gbm_team | +0.0311 | 0.9146 / 0.947 / 0.925 | 0.8088 / 0.776 / 0.921 | 0.9278 / 0.977 / 0.942 | 0.8236 / 0.789 / 0.938 |
| gbm_specialists | +0.0279 | 0.9133 / 0.958 / 0.923 | 0.8070 / 0.775 / 0.917 | 0.9248 / 0.983 / 0.934 | 0.8100 / 0.785 / 0.919 |
| gbm_behavior_aware | +0.0260 | 0.9046 / 0.946 / 0.914 | 0.8043 / 0.768 / 0.913 | 0.9265 / 0.980 / 0.938 | 0.8223 / 0.795 / 0.939 |
| gbm_standard | +0.0119 | 0.8986 / 0.905 / 0.932 | 0.8004 / 0.763 / 0.934 | 0.9165 / 0.930 / 0.962 | 0.8185 / 0.773 / 0.963 |
| proxy_gbm | +0.0023 | 0.8975 / 0.905 / 0.917 | 0.7856 / 0.757 / 0.904 | 0.9059 / 0.919 / 0.931 | 0.8042 / 0.778 / 0.932 |
| reference | +0.0000 | 0.8970 / 0.893 / 0.913 | 0.7871 / 0.731 / 0.896 | 0.9059 / 0.919 / 0.931 | 0.8042 / 0.778 / 0.932 |
| lr_team | -0.0064 | 0.8928 / 0.867 / 0.904 | 0.7785 / 0.710 / 0.876 | 0.9059 / 0.919 / 0.931 | 0.8042 / 0.778 / 0.932 |
| handrule_paper | -0.0224 | 0.8608 / 0.864 / 0.876 | 0.7659 / 0.735 / 0.875 | 0.8987 / 0.918 / 0.937 | 0.7999 / 0.769 / 0.937 |
| handrule_score_heuristic | -0.0241 | 0.8542 / 0.869 / 0.867 | 0.7606 / 0.737 / 0.868 | 0.9001 / 0.926 / 0.936 | 0.8006 / 0.771 / 0.938 |
| rrs | -0.0558 | 0.8023 / 0.803 / 0.782 | 0.7309 / 0.659 / 0.786 | — | — |
| behavior_aware_ai | -0.1098 | 0.8084 / 0.778 / 0.782 | 0.7311 / 0.656 / 0.786 | 0.8095 / 0.774 / 0.784 | 0.7327 / 0.652 / 0.784 |
| standard_ai | -0.1133 | 0.8075 / 0.775 / 0.786 | 0.7184 / 0.690 / 0.786 | 0.8062 / 0.768 / 0.786 | 0.7181 / 0.692 / 0.785 |
| complementary_ai | -0.1248 | 0.8335 / 0.596 / 0.781 | 0.7480 / 0.576 / 0.784 | 0.8316 / 0.591 / 0.776 | 0.7479 / 0.564 / 0.775 |
| aligned_ai | -0.1533 | 0.7346 / 0.920 / 0.742 | 0.6506 / 0.774 / 0.737 | 0.7343 / 0.919 / 0.744 | 0.6504 / 0.778 / 0.734 |
| aligned_everywhere_gbm | -0.1571 | 0.7187 / 0.971 / 0.726 | 0.6387 / 0.779 / 0.705 | 0.7186 / 0.983 / 0.724 | 0.6431 / 0.795 / 0.713 |

## world `no_red_flag`

- admission_group_hidden: human acc privileged 0.895, underprivileged 0.499, privileged share 0.501, mean P(alignment) 0.467, slip 0.054
- admission_group_hidden_low_reliability: human acc privileged 0.750, underprivileged 0.508, privileged share 0.501, mean P(alignment) 0.405, slip 0.221
- admission_group_known: human acc privileged 0.894, underprivileged 0.507, privileged share 0.503, mean P(alignment) 0.469, slip 0.054
- admission_group_known_low_reliability: human acc privileged 0.750, underprivileged 0.505, privileged share 0.500, mean P(alignment) 0.402, slip 0.221

| method | raw gain | admission_group_hidden team / rel / ai | admission_group_hidden_low_reliability team / rel / ai | admission_group_known team / rel / ai | admission_group_known_low_reliability team / rel / ai |
|---|---|---|---|---|---|
| generator_bayes_policy | +0.0243 | 0.9169 / 0.907 / 0.960 | 0.8253 / 0.761 / 0.963 | 0.9298 / 0.930 / 0.978 | 0.8356 / 0.781 / 0.979 |
| gbm_team | +0.0071 | 0.9010 / 0.905 / 0.935 | 0.8020 / 0.743 / 0.925 | 0.9196 / 0.936 / 0.962 | 0.8208 / 0.765 / 0.954 |
| gbm_standard | +0.0061 | 0.8938 / 0.903 / 0.929 | 0.8006 / 0.762 / 0.930 | 0.9188 / 0.933 / 0.962 | 0.8240 / 0.781 / 0.964 |
| gbm_behavior_aware | +0.0040 | 0.8942 / 0.904 / 0.926 | 0.7984 / 0.742 / 0.917 | 0.9176 / 0.939 / 0.957 | 0.8211 / 0.774 / 0.957 |
| gbm_specialists | +0.0027 | 0.8981 / 0.913 / 0.931 | 0.8003 / 0.743 / 0.922 | 0.9160 / 0.940 / 0.953 | 0.8061 / 0.762 / 0.932 |
| proxy_gbm | +0.0012 | 0.8968 / 0.913 / 0.929 | 0.7849 / 0.753 / 0.906 | 0.9150 / 0.942 / 0.952 | 0.8097 / 0.781 / 0.945 |
| reference | +0.0000 | 0.8966 / 0.902 / 0.926 | 0.7894 / 0.731 / 0.900 | 0.9150 / 0.942 / 0.952 | 0.8097 / 0.781 / 0.945 |
| lr_team | -0.0092 | 0.8904 / 0.865 / 0.912 | 0.7774 / 0.704 / 0.874 | 0.9150 / 0.942 / 0.952 | 0.8097 / 0.781 / 0.945 |
| handrule_paper | -0.0274 | 0.8581 / 0.863 / 0.875 | 0.7661 / 0.736 / 0.873 | 0.9009 / 0.921 / 0.938 | 0.8040 / 0.779 / 0.938 |
| handrule_score_heuristic | -0.0300 | 0.8507 / 0.868 / 0.865 | 0.7606 / 0.737 / 0.866 | 0.9013 / 0.928 / 0.936 | 0.8042 / 0.780 / 0.938 |
| rrs | -0.0585 | 0.8041 / 0.776 / 0.786 | 0.7327 / 0.657 / 0.786 | — | — |
| behavior_aware_ai | -0.1180 | 0.8054 / 0.772 / 0.786 | 0.7332 / 0.653 / 0.786 | 0.8094 / 0.765 / 0.785 | 0.7324 / 0.648 / 0.785 |
| standard_ai | -0.1191 | 0.8042 / 0.776 / 0.786 | 0.7198 / 0.692 / 0.786 | 0.8078 / 0.772 / 0.785 | 0.7187 / 0.696 / 0.784 |
| complementary_ai | -0.1316 | 0.8315 / 0.595 / 0.782 | 0.7494 / 0.583 / 0.785 | 0.8311 / 0.585 / 0.776 | 0.7478 / 0.563 / 0.777 |
| aligned_ai | -0.1526 | 0.7346 / 0.938 / 0.761 | 0.6560 / 0.776 / 0.750 | 0.7404 / 0.942 / 0.766 | 0.6553 / 0.781 / 0.747 |
| aligned_everywhere_gbm | -0.1751 | 0.7069 / 0.929 / 0.728 | 0.6436 / 0.755 / 0.719 | 0.7182 / 0.941 / 0.743 | 0.6428 / 0.774 / 0.727 |

