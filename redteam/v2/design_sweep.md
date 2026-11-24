# World-design sweep (team accuracy / reliance on each world's hidden test)

## world `default`

- admission_group_hidden: human acc privileged 0.898, underprivileged 0.504, privileged share 0.499, mean P(alignment) 0.485, slip 0.015
- admission_group_hidden_low_reliability: human acc privileged 0.753, underprivileged 0.505, privileged share 0.495, mean P(alignment) 0.454, slip 0.197
- admission_group_known: human acc privileged 0.898, underprivileged 0.508, privileged share 0.498, mean P(alignment) 0.484, slip 0.015
- admission_group_known_low_reliability: human acc privileged 0.752, underprivileged 0.502, privileged share 0.499, mean P(alignment) 0.457, slip 0.197

| method | raw gain | admission_group_hidden team / rel / ai | admission_group_hidden_low_reliability team / rel / ai | admission_group_known team / rel / ai | admission_group_known_low_reliability team / rel / ai |
|---|---|---|---|---|---|
| generator_bayes_policy | +0.0307 | 0.9237 / 0.950 / 0.937 | 0.8225 / 0.780 / 0.948 | 0.9392 / 0.976 / 0.956 | 0.8304 / 0.796 / 0.967 |
| hybrid_comp_gbm_router_gbm | +0.0134 | 0.9050 / 0.954 / 0.912 | 0.7991 / 0.779 / 0.917 | 0.9254 / 0.981 / 0.936 | 0.8138 / 0.797 / 0.943 |
| gbm_team | +0.0072 | 0.9060 / 0.934 / 0.916 | 0.7972 / 0.755 / 0.912 | 0.9242 / 0.969 / 0.941 | 0.8105 / 0.769 / 0.943 |
| gbm_specialists | +0.0070 | 0.9044 / 0.944 / 0.916 | 0.7946 / 0.765 / 0.906 | 0.9235 / 0.971 / 0.938 | 0.8070 / 0.776 / 0.930 |
| hybrid_comp_gbm | +0.0070 | 0.8958 / 0.938 / 0.899 | 0.7946 / 0.762 / 0.903 | 0.9254 / 0.981 / 0.936 | 0.8138 / 0.797 / 0.943 |
| proxy_gbm | +0.0060 | 0.8992 / 0.954 / 0.906 | 0.7898 / 0.781 / 0.903 | 0.9174 / 0.981 / 0.928 | 0.8062 / 0.797 / 0.934 |
| gbm_behavior_aware | +0.0013 | 0.8941 / 0.926 / 0.905 | 0.7917 / 0.750 / 0.901 | 0.9238 / 0.968 / 0.942 | 0.8108 / 0.777 / 0.947 |
| reference | +0.0000 | 0.8904 / 0.939 / 0.893 | 0.7863 / 0.763 / 0.891 | 0.9174 / 0.981 / 0.928 | 0.8062 / 0.797 / 0.934 |
| lr_team | -0.0051 | 0.8907 / 0.907 / 0.889 | 0.7815 / 0.734 / 0.872 | 0.9174 / 0.981 / 0.928 | 0.8062 / 0.797 / 0.934 |
| gbm_standard | -0.0054 | 0.8881 / 0.897 / 0.919 | 0.7925 / 0.766 / 0.920 | 0.9173 / 0.930 / 0.960 | 0.8147 / 0.781 / 0.960 |
| handrule_paper | -0.0430 | 0.8453 / 0.875 / 0.850 | 0.7541 / 0.747 / 0.851 | 0.8881 / 0.935 / 0.914 | 0.7885 / 0.783 / 0.916 |
| handrule_score_heuristic | -0.0536 | 0.8350 / 0.855 / 0.840 | 0.7475 / 0.738 / 0.842 | 0.8840 / 0.913 / 0.912 | 0.7868 / 0.770 / 0.916 |
| rrs | -0.0594 | 0.7950 / 0.825 / 0.765 | 0.7350 / 0.646 / 0.770 | — | — |
| behavior_aware_ai | -0.1205 | 0.8042 / 0.797 / 0.768 | 0.7300 / 0.664 / 0.769 | 0.8079 / 0.786 / 0.765 | 0.7316 / 0.652 / 0.768 |
| standard_ai | -0.1236 | 0.8029 / 0.794 / 0.769 | 0.7175 / 0.696 / 0.770 | 0.8054 / 0.789 / 0.767 | 0.7151 / 0.697 / 0.766 |
| complementary_ai | -0.1382 | 0.8315 / 0.601 / 0.764 | 0.7461 / 0.569 / 0.760 | 0.8298 / 0.589 / 0.755 | 0.7459 / 0.560 / 0.754 |
| aligned_ai | -0.1544 | 0.7307 / 0.979 / 0.738 | 0.6505 / 0.796 / 0.736 | 0.7364 / 0.981 / 0.744 | 0.6497 / 0.797 / 0.737 |
| aligned_everywhere_gbm | -0.1743 | 0.7073 / 0.952 / 0.714 | 0.6424 / 0.773 / 0.713 | 0.7236 / 0.969 / 0.733 | 0.6390 / 0.789 / 0.719 |

## world `under_mild`

- admission_group_hidden: human acc privileged 0.898, underprivileged 0.503, privileged share 0.499, mean P(alignment) 0.485, slip 0.015
- admission_group_hidden_low_reliability: human acc privileged 0.753, underprivileged 0.504, privileged share 0.495, mean P(alignment) 0.454, slip 0.197
- admission_group_known: human acc privileged 0.898, underprivileged 0.507, privileged share 0.498, mean P(alignment) 0.484, slip 0.015
- admission_group_known_low_reliability: human acc privileged 0.752, underprivileged 0.505, privileged share 0.499, mean P(alignment) 0.457, slip 0.197

| method | raw gain | admission_group_hidden team / rel / ai | admission_group_hidden_low_reliability team / rel / ai | admission_group_known team / rel / ai | admission_group_known_low_reliability team / rel / ai |
|---|---|---|---|---|---|
| generator_bayes_policy | +0.0267 | 0.9240 / 0.951 / 0.937 | 0.8229 / 0.780 / 0.949 | 0.9392 / 0.976 / 0.956 | 0.8307 / 0.796 / 0.967 |
| hybrid_comp_gbm_router_gbm | +0.0113 | 0.9076 / 0.954 / 0.914 | 0.8008 / 0.780 / 0.920 | 0.9274 / 0.981 / 0.939 | 0.8158 / 0.797 / 0.945 |
| proxy_gbm | +0.0061 | 0.9035 / 0.955 / 0.910 | 0.7936 / 0.781 / 0.908 | 0.9228 / 0.981 / 0.934 | 0.8099 / 0.797 / 0.938 |
| hybrid_comp_gbm | +0.0050 | 0.8983 / 0.939 / 0.902 | 0.7964 / 0.764 / 0.906 | 0.9274 / 0.981 / 0.939 | 0.8158 / 0.797 / 0.945 |
| gbm_team | +0.0050 | 0.9087 / 0.935 / 0.919 | 0.7990 / 0.758 / 0.915 | 0.9253 / 0.969 / 0.942 | 0.8120 / 0.770 / 0.945 |
| gbm_specialists | +0.0049 | 0.9069 / 0.944 / 0.918 | 0.7962 / 0.766 / 0.908 | 0.9255 / 0.971 / 0.940 | 0.8090 / 0.776 / 0.932 |
| reference | +0.0000 | 0.8943 / 0.940 / 0.898 | 0.7899 / 0.764 / 0.896 | 0.9228 / 0.981 / 0.934 | 0.8099 / 0.797 / 0.938 |
| gbm_behavior_aware | -0.0001 | 0.8970 / 0.928 / 0.908 | 0.7950 / 0.754 / 0.906 | 0.9248 / 0.967 / 0.943 | 0.8140 / 0.778 / 0.951 |
| lr_team | -0.0051 | 0.8946 / 0.908 / 0.893 | 0.7850 / 0.735 / 0.877 | 0.9228 / 0.981 / 0.934 | 0.8099 / 0.797 / 0.938 |
| gbm_standard | -0.0079 | 0.8898 / 0.896 / 0.921 | 0.7952 / 0.766 / 0.924 | 0.9187 / 0.930 / 0.961 | 0.8169 / 0.781 / 0.962 |
| handrule_paper | -0.0384 | 0.8537 / 0.875 / 0.860 | 0.7618 / 0.747 / 0.861 | 0.8987 / 0.935 / 0.925 | 0.7976 / 0.783 / 0.927 |
| handrule_score_heuristic | -0.0491 | 0.8431 / 0.855 / 0.850 | 0.7551 / 0.738 / 0.853 | 0.8944 / 0.913 / 0.924 | 0.7958 / 0.770 / 0.927 |
| rrs | -0.0587 | 0.7997 / 0.829 / 0.771 | 0.7386 / 0.650 / 0.776 | — | — |
| behavior_aware_ai | -0.1190 | 0.8085 / 0.801 / 0.774 | 0.7347 / 0.667 / 0.777 | 0.8132 / 0.794 / 0.773 | 0.7360 / 0.654 / 0.776 |
| standard_ai | -0.1223 | 0.8064 / 0.801 / 0.776 | 0.7228 / 0.698 / 0.778 | 0.8102 / 0.797 / 0.775 | 0.7197 / 0.699 / 0.775 |
| complementary_ai | -0.1348 | 0.8366 / 0.613 / 0.774 | 0.7500 / 0.576 / 0.768 | 0.8363 / 0.604 / 0.768 | 0.7503 / 0.567 / 0.763 |
| aligned_ai | -0.1510 | 0.7382 / 0.979 / 0.746 | 0.6575 / 0.796 / 0.745 | 0.7454 / 0.981 / 0.753 | 0.6569 / 0.797 / 0.746 |
| aligned_everywhere_gbm | -0.1772 | 0.7107 / 0.952 / 0.718 | 0.6426 / 0.773 / 0.713 | 0.7237 / 0.969 / 0.733 | 0.6399 / 0.789 / 0.720 |

## world `under_strong`

- admission_group_hidden: human acc privileged 0.898, underprivileged 0.502, privileged share 0.499, mean P(alignment) 0.485, slip 0.015
- admission_group_hidden_low_reliability: human acc privileged 0.753, underprivileged 0.507, privileged share 0.495, mean P(alignment) 0.454, slip 0.197
- admission_group_known: human acc privileged 0.898, underprivileged 0.507, privileged share 0.498, mean P(alignment) 0.484, slip 0.015
- admission_group_known_low_reliability: human acc privileged 0.752, underprivileged 0.501, privileged share 0.499, mean P(alignment) 0.457, slip 0.197

| method | raw gain | admission_group_hidden team / rel / ai | admission_group_hidden_low_reliability team / rel / ai | admission_group_known team / rel / ai | admission_group_known_low_reliability team / rel / ai |
|---|---|---|---|---|---|
| generator_bayes_policy | +0.0369 | 0.9230 / 0.949 / 0.936 | 0.8223 / 0.778 / 0.946 | 0.9392 / 0.976 / 0.956 | 0.8303 / 0.796 / 0.967 |
| hybrid_comp_gbm_router_gbm | +0.0182 | 0.9032 / 0.953 / 0.910 | 0.7953 / 0.780 / 0.911 | 0.9249 / 0.981 / 0.936 | 0.8122 / 0.797 / 0.941 |
| gbm_specialists | +0.0119 | 0.9026 / 0.942 / 0.914 | 0.7909 / 0.766 / 0.900 | 0.9230 / 0.971 / 0.938 | 0.8054 / 0.776 / 0.928 |
| hybrid_comp_gbm | +0.0115 | 0.8938 / 0.935 / 0.897 | 0.7908 / 0.761 / 0.897 | 0.9249 / 0.981 / 0.936 | 0.8122 / 0.797 / 0.941 |
| gbm_team | +0.0112 | 0.9043 / 0.932 / 0.914 | 0.7930 / 0.753 / 0.906 | 0.9226 / 0.969 / 0.939 | 0.8074 / 0.769 / 0.940 |
| proxy_gbm | +0.0060 | 0.8907 / 0.953 / 0.897 | 0.7847 / 0.781 / 0.896 | 0.9108 / 0.981 / 0.922 | 0.8009 / 0.797 / 0.927 |
| gbm_behavior_aware | +0.0058 | 0.8913 / 0.925 / 0.901 | 0.7891 / 0.749 / 0.895 | 0.9231 / 0.968 / 0.942 | 0.8090 / 0.776 / 0.945 |
| reference | +0.0000 | 0.8823 / 0.937 / 0.884 | 0.7814 / 0.762 / 0.884 | 0.9108 / 0.981 / 0.922 | 0.8009 / 0.797 / 0.927 |
| gbm_standard | -0.0008 | 0.8859 / 0.896 / 0.916 | 0.7880 / 0.763 / 0.914 | 0.9164 / 0.931 / 0.958 | 0.8142 / 0.781 / 0.959 |
| lr_team | -0.0052 | 0.8833 / 0.905 / 0.880 | 0.7761 / 0.729 / 0.863 | 0.9108 / 0.981 / 0.922 | 0.8009 / 0.797 / 0.927 |
| handrule_paper | -0.0481 | 0.8337 / 0.875 / 0.837 | 0.7437 / 0.747 / 0.836 | 0.8749 / 0.935 / 0.900 | 0.7767 / 0.783 / 0.901 |
| handrule_score_heuristic | -0.0583 | 0.8237 / 0.855 / 0.827 | 0.7373 / 0.738 / 0.828 | 0.8711 / 0.913 / 0.898 | 0.7752 / 0.770 / 0.901 |
| rrs | -0.0594 | 0.7888 / 0.820 / 0.756 | 0.7305 / 0.636 / 0.758 | — | — |
| behavior_aware_ai | -0.1214 | 0.7976 / 0.791 / 0.758 | 0.7249 / 0.658 / 0.758 | 0.8008 / 0.782 / 0.755 | 0.7258 / 0.647 / 0.757 |
| standard_ai | -0.1246 | 0.7972 / 0.788 / 0.761 | 0.7120 / 0.693 / 0.759 | 0.7972 / 0.784 / 0.756 | 0.7081 / 0.693 / 0.756 |
| complementary_ai | -0.1420 | 0.8223 / 0.582 / 0.749 | 0.7406 / 0.558 / 0.746 | 0.8232 / 0.576 / 0.745 | 0.7404 / 0.551 / 0.742 |
| aligned_ai | -0.1578 | 0.7199 / 0.979 / 0.727 | 0.6413 / 0.796 / 0.724 | 0.7254 / 0.981 / 0.733 | 0.6411 / 0.797 / 0.727 |
| aligned_everywhere_gbm | -0.1697 | 0.7029 / 0.952 / 0.710 | 0.6415 / 0.773 / 0.712 | 0.7222 / 0.969 / 0.732 | 0.6389 / 0.789 / 0.719 |

## world `under_narrow_gate`

- admission_group_hidden: human acc privileged 0.897, underprivileged 0.503, privileged share 0.501, mean P(alignment) 0.487, slip 0.015
- admission_group_hidden_low_reliability: human acc privileged 0.752, underprivileged 0.506, privileged share 0.501, mean P(alignment) 0.460, slip 0.197
- admission_group_known: human acc privileged 0.899, underprivileged 0.508, privileged share 0.503, mean P(alignment) 0.488, slip 0.015
- admission_group_known_low_reliability: human acc privileged 0.753, underprivileged 0.502, privileged share 0.500, mean P(alignment) 0.457, slip 0.197

| method | raw gain | admission_group_hidden team / rel / ai | admission_group_hidden_low_reliability team / rel / ai | admission_group_known team / rel / ai | admission_group_known_low_reliability team / rel / ai |
|---|---|---|---|---|---|
| generator_bayes_policy | +0.0256 | 0.9231 / 0.947 / 0.937 | 0.8212 / 0.777 / 0.947 | 0.9388 / 0.976 / 0.956 | 0.8310 / 0.797 / 0.967 |
| hybrid_comp_gbm_router_gbm | +0.0092 | 0.9050 / 0.954 / 0.913 | 0.7987 / 0.777 / 0.918 | 0.9247 / 0.981 / 0.936 | 0.8151 / 0.799 / 0.944 |
| hybrid_comp_gbm | +0.0071 | 0.9025 / 0.948 / 0.908 | 0.7989 / 0.764 / 0.911 | 0.9247 / 0.981 / 0.936 | 0.8151 / 0.799 / 0.944 |
| gbm_team | +0.0033 | 0.9063 / 0.935 / 0.917 | 0.7969 / 0.751 / 0.914 | 0.9237 / 0.970 / 0.940 | 0.8121 / 0.772 / 0.945 |
| gbm_specialists | +0.0029 | 0.9045 / 0.944 / 0.916 | 0.7932 / 0.761 / 0.906 | 0.9233 / 0.971 / 0.939 | 0.8087 / 0.778 / 0.932 |
| proxy_gbm | +0.0019 | 0.8987 / 0.955 / 0.906 | 0.7898 / 0.780 / 0.905 | 0.9168 / 0.981 / 0.928 | 0.8073 / 0.799 / 0.934 |
| reference | +0.0000 | 0.8967 / 0.950 / 0.902 | 0.7902 / 0.765 / 0.899 | 0.9168 / 0.981 / 0.928 | 0.8073 / 0.799 / 0.934 |
| gbm_behavior_aware | -0.0015 | 0.8963 / 0.933 / 0.910 | 0.7932 / 0.749 / 0.903 | 0.9221 / 0.966 / 0.941 | 0.8121 / 0.777 / 0.948 |
| lr_team | -0.0057 | 0.8960 / 0.922 / 0.897 | 0.7828 / 0.736 / 0.876 | 0.9168 / 0.981 / 0.928 | 0.8073 / 0.799 / 0.934 |
| gbm_standard | -0.0080 | 0.8907 / 0.901 / 0.924 | 0.7946 / 0.764 / 0.925 | 0.9164 / 0.929 / 0.960 | 0.8163 / 0.781 / 0.962 |
| handrule_paper | -0.0460 | 0.8474 / 0.878 / 0.854 | 0.7553 / 0.745 / 0.853 | 0.8881 / 0.935 / 0.914 | 0.7890 / 0.784 / 0.916 |
| handrule_score_heuristic | -0.0562 | 0.8377 / 0.859 / 0.845 | 0.7492 / 0.736 / 0.846 | 0.8839 / 0.912 / 0.912 | 0.7873 / 0.770 / 0.916 |
| rrs | -0.0636 | 0.7950 / 0.825 / 0.765 | 0.7351 / 0.643 / 0.769 | — | — |
| behavior_aware_ai | -0.1245 | 0.8040 / 0.799 / 0.768 | 0.7297 / 0.663 / 0.769 | 0.8087 / 0.786 / 0.765 | 0.7317 / 0.651 / 0.767 |
| standard_ai | -0.1278 | 0.8019 / 0.798 / 0.769 | 0.7171 / 0.696 / 0.769 | 0.8055 / 0.791 / 0.767 | 0.7149 / 0.698 / 0.766 |
| complementary_ai | -0.1424 | 0.8308 / 0.600 / 0.763 | 0.7467 / 0.569 / 0.760 | 0.8307 / 0.590 / 0.754 | 0.7461 / 0.558 / 0.753 |
| aligned_ai | -0.1569 | 0.7317 / 0.979 / 0.739 | 0.6520 / 0.795 / 0.739 | 0.7382 / 0.981 / 0.746 | 0.6526 / 0.799 / 0.741 |
| aligned_everywhere_gbm | -0.1759 | 0.7106 / 0.953 / 0.718 | 0.6445 / 0.773 / 0.717 | 0.7240 / 0.968 / 0.733 | 0.6437 / 0.787 / 0.724 |

