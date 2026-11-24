# World-design sweep (team accuracy / reliance on each world's hidden test)

## world `default`

- admission_group_hidden: human acc privileged 0.897, underprivileged 0.499, privileged share 0.501, mean P(alignment) 0.487, slip 0.015
- admission_group_hidden_low_reliability: human acc privileged 0.752, underprivileged 0.506, privileged share 0.501, mean P(alignment) 0.460, slip 0.197
- admission_group_known: human acc privileged 0.899, underprivileged 0.508, privileged share 0.503, mean P(alignment) 0.488, slip 0.015
- admission_group_known_low_reliability: human acc privileged 0.753, underprivileged 0.504, privileged share 0.500, mean P(alignment) 0.457, slip 0.197

| method | raw gain | admission_group_hidden team / rel / ai | admission_group_hidden_low_reliability team / rel / ai | admission_group_known team / rel / ai | admission_group_known_low_reliability team / rel / ai |
|---|---|---|---|---|---|
| generator_bayes_policy | +0.0176 | 0.9241 / 0.949 / 0.938 | 0.8220 / 0.779 / 0.949 | 0.9388 / 0.976 / 0.956 | 0.8312 / 0.797 / 0.967 |
| proxy_gbm | +0.0022 | 0.9087 / 0.956 / 0.917 | 0.7967 / 0.781 / 0.914 | 0.9265 / 0.981 / 0.938 | 0.8160 / 0.799 / 0.945 |
| reference | +0.0000 | 0.9057 / 0.951 / 0.912 | 0.7969 / 0.767 / 0.909 | 0.9265 / 0.981 / 0.938 | 0.8160 / 0.799 / 0.945 |
| gbm_team | -0.0019 | 0.9105 / 0.937 / 0.922 | 0.7994 / 0.755 / 0.919 | 0.9263 / 0.970 / 0.943 | 0.8150 / 0.773 / 0.949 |
| gbm_specialists | -0.0023 | 0.9091 / 0.946 / 0.921 | 0.7956 / 0.762 / 0.910 | 0.9269 / 0.971 / 0.942 | 0.8112 / 0.778 / 0.935 |
| lr_team | -0.0052 | 0.9050 / 0.925 / 0.908 | 0.7901 / 0.741 / 0.888 | 0.9265 / 0.981 / 0.938 | 0.8160 / 0.799 / 0.945 |
| gbm_behavior_aware | -0.0062 | 0.8990 / 0.934 / 0.912 | 0.7977 / 0.755 / 0.913 | 0.9258 / 0.966 / 0.944 | 0.8165 / 0.780 / 0.953 |
| gbm_standard | -0.0133 | 0.8944 / 0.903 / 0.929 | 0.7980 / 0.767 / 0.930 | 0.9195 / 0.929 / 0.962 | 0.8184 / 0.781 / 0.964 |
| handrule_paper | -0.0367 | 0.8650 / 0.878 / 0.875 | 0.7702 / 0.745 / 0.873 | 0.9103 / 0.935 / 0.938 | 0.8064 / 0.784 / 0.938 |
| handrule_score_heuristic | -0.0473 | 0.8549 / 0.859 / 0.865 | 0.7640 / 0.736 / 0.866 | 0.9055 / 0.912 / 0.936 | 0.8044 / 0.770 / 0.938 |
| rrs | -0.0615 | 0.8053 / 0.836 / 0.781 | 0.7424 / 0.658 / 0.787 | — | — |
| behavior_aware_ai | -0.1208 | 0.8143 / 0.808 / 0.784 | 0.7382 / 0.672 / 0.786 | 0.8203 / 0.801 / 0.784 | 0.7400 / 0.663 / 0.785 |
| standard_ai | -0.1238 | 0.8118 / 0.810 / 0.786 | 0.7261 / 0.702 / 0.786 | 0.8171 / 0.806 / 0.785 | 0.7248 / 0.707 / 0.784 |
| complementary_ai | -0.1353 | 0.8414 / 0.625 / 0.783 | 0.7544 / 0.585 / 0.779 | 0.8424 / 0.617 / 0.776 | 0.7568 / 0.578 / 0.777 |
| aligned_ai | -0.1491 | 0.7488 / 0.979 / 0.757 | 0.6664 / 0.795 / 0.757 | 0.7570 / 0.981 / 0.765 | 0.6681 / 0.799 / 0.760 |
| aligned_everywhere_gbm | -0.1818 | 0.7175 / 0.953 / 0.725 | 0.6467 / 0.773 / 0.720 | 0.7236 / 0.968 / 0.733 | 0.6451 / 0.787 / 0.725 |

## world `bias_wide_gate`

- admission_group_hidden: human acc privileged 0.898, underprivileged 0.501, privileged share 0.499, mean P(alignment) 0.485, slip 0.015
- admission_group_hidden_low_reliability: human acc privileged 0.753, underprivileged 0.504, privileged share 0.495, mean P(alignment) 0.454, slip 0.197
- admission_group_known: human acc privileged 0.898, underprivileged 0.507, privileged share 0.498, mean P(alignment) 0.484, slip 0.015
- admission_group_known_low_reliability: human acc privileged 0.752, underprivileged 0.505, privileged share 0.499, mean P(alignment) 0.457, slip 0.197

| method | raw gain | admission_group_hidden team / rel / ai | admission_group_hidden_low_reliability team / rel / ai | admission_group_known team / rel / ai | admission_group_known_low_reliability team / rel / ai |
|---|---|---|---|---|---|
| generator_bayes_policy | +0.0222 | 0.9242 / 0.951 / 0.938 | 0.8234 / 0.782 / 0.950 | 0.9392 / 0.976 / 0.956 | 0.8307 / 0.796 / 0.967 |
| proxy_gbm | +0.0060 | 0.9083 / 0.955 / 0.915 | 0.7975 / 0.782 / 0.914 | 0.9275 / 0.981 / 0.939 | 0.8148 / 0.797 / 0.944 |
| gbm_team | +0.0015 | 0.9096 / 0.935 / 0.920 | 0.7998 / 0.759 / 0.917 | 0.9268 / 0.969 / 0.944 | 0.8135 / 0.769 / 0.947 |
| gbm_specialists | +0.0013 | 0.9085 / 0.945 / 0.920 | 0.7971 / 0.766 / 0.910 | 0.9267 / 0.971 / 0.942 | 0.8096 / 0.776 / 0.933 |
| reference | +0.0000 | 0.8992 / 0.940 / 0.903 | 0.7938 / 0.766 / 0.902 | 0.9275 / 0.981 / 0.939 | 0.8148 / 0.797 / 0.944 |
| gbm_behavior_aware | -0.0038 | 0.8966 / 0.928 / 0.908 | 0.7960 / 0.754 / 0.909 | 0.9267 / 0.968 / 0.944 | 0.8164 / 0.778 / 0.953 |
| lr_team | -0.0048 | 0.8996 / 0.910 / 0.899 | 0.7889 / 0.739 / 0.884 | 0.9275 / 0.981 / 0.939 | 0.8148 / 0.797 / 0.944 |
| gbm_standard | -0.0115 | 0.8905 / 0.897 / 0.922 | 0.7960 / 0.768 / 0.925 | 0.9204 / 0.930 / 0.963 | 0.8174 / 0.781 / 0.963 |
| handrule_paper | -0.0339 | 0.8626 / 0.875 / 0.870 | 0.7695 / 0.747 / 0.871 | 0.9103 / 0.935 / 0.937 | 0.8060 / 0.783 / 0.938 |
| handrule_score_heuristic | -0.0448 | 0.8519 / 0.855 / 0.860 | 0.7627 / 0.738 / 0.863 | 0.9057 / 0.913 / 0.936 | 0.8041 / 0.770 / 0.938 |
| rrs | -0.0571 | 0.8064 / 0.837 / 0.782 | 0.7424 / 0.659 / 0.787 | — | — |
| behavior_aware_ai | -0.1168 | 0.8149 / 0.807 / 0.784 | 0.7389 / 0.672 / 0.787 | 0.8191 / 0.801 / 0.782 | 0.7404 / 0.662 / 0.786 |
| standard_ai | -0.1199 | 0.8124 / 0.807 / 0.785 | 0.7271 / 0.702 / 0.787 | 0.8166 / 0.804 / 0.785 | 0.7253 / 0.705 / 0.785 |
| complementary_ai | -0.1311 | 0.8417 / 0.624 / 0.784 | 0.7546 / 0.586 / 0.780 | 0.8421 / 0.618 / 0.778 | 0.7566 / 0.580 / 0.777 |
| aligned_ai | -0.1467 | 0.7477 / 0.979 / 0.755 | 0.6653 / 0.796 / 0.755 | 0.7553 / 0.981 / 0.763 | 0.6653 / 0.797 / 0.756 |
| aligned_everywhere_gbm | -0.1804 | 0.7140 / 0.952 / 0.721 | 0.6439 / 0.773 / 0.715 | 0.7240 / 0.969 / 0.733 | 0.6406 / 0.789 / 0.721 |

## world `bias_strong`

- admission_group_hidden: human acc privileged 0.893, underprivileged 0.499, privileged share 0.501, mean P(alignment) 0.488, slip 0.000
- admission_group_hidden_low_reliability: human acc privileged 0.753, underprivileged 0.505, privileged share 0.501, mean P(alignment) 0.460, slip 0.180
- admission_group_known: human acc privileged 0.888, underprivileged 0.509, privileged share 0.503, mean P(alignment) 0.489, slip 0.000
- admission_group_known_low_reliability: human acc privileged 0.753, underprivileged 0.504, privileged share 0.500, mean P(alignment) 0.456, slip 0.180

| method | raw gain | admission_group_hidden team / rel / ai | admission_group_hidden_low_reliability team / rel / ai | admission_group_known team / rel / ai | admission_group_known_low_reliability team / rel / ai |
|---|---|---|---|---|---|
| generator_bayes_policy | +0.0172 | 0.9257 / 0.967 / 0.927 | 0.8264 / 0.795 / 0.940 | 0.9371 / 0.990 / 0.944 | 0.8366 / 0.813 / 0.958 |
| proxy_gbm | +0.0020 | 0.9104 / 0.973 / 0.909 | 0.8016 / 0.795 / 0.908 | 0.9250 / 0.998 / 0.926 | 0.8214 / 0.817 / 0.934 |
| reference | +0.0000 | 0.9078 / 0.967 / 0.905 | 0.8018 / 0.784 / 0.903 | 0.9250 / 0.998 / 0.926 | 0.8214 / 0.817 / 0.934 |
| gbm_team | -0.0046 | 0.9102 / 0.951 / 0.913 | 0.8034 / 0.766 / 0.916 | 0.9249 / 0.983 / 0.933 | 0.8176 / 0.783 / 0.942 |
| gbm_specialists | -0.0047 | 0.9086 / 0.960 / 0.913 | 0.7999 / 0.774 / 0.905 | 0.9249 / 0.984 / 0.932 | 0.8143 / 0.791 / 0.925 |
| lr_team | -0.0055 | 0.9069 / 0.944 / 0.901 | 0.7932 / 0.754 / 0.882 | 0.9250 / 0.998 / 0.926 | 0.8214 / 0.817 / 0.934 |
| gbm_behavior_aware | -0.0087 | 0.8987 / 0.947 / 0.905 | 0.8007 / 0.767 / 0.908 | 0.9242 / 0.980 / 0.935 | 0.8204 / 0.792 / 0.947 |
| gbm_standard | -0.0230 | 0.8910 / 0.898 / 0.929 | 0.7994 / 0.768 / 0.930 | 0.9124 / 0.918 / 0.962 | 0.8210 / 0.784 / 0.964 |
| handrule_paper | -0.0459 | 0.8620 / 0.872 / 0.875 | 0.7728 / 0.748 / 0.873 | 0.9033 / 0.921 / 0.938 | 0.8100 / 0.787 / 0.938 |
| handrule_score_heuristic | -0.0553 | 0.8524 / 0.856 / 0.865 | 0.7663 / 0.740 / 0.866 | 0.8996 / 0.904 / 0.936 | 0.8078 / 0.776 / 0.938 |
| rrs | -0.0636 | 0.7950 / 0.886 / 0.777 | 0.7453 / 0.669 / 0.787 | — | — |
| behavior_aware_ai | -0.1247 | 0.8119 / 0.824 / 0.782 | 0.7421 / 0.686 / 0.787 | 0.8154 / 0.812 / 0.782 | 0.7429 / 0.675 / 0.785 |
| standard_ai | -0.1293 | 0.8105 / 0.815 / 0.786 | 0.7295 / 0.711 / 0.786 | 0.8126 / 0.806 / 0.785 | 0.7282 / 0.714 / 0.784 |
| complementary_ai | -0.1416 | 0.8404 / 0.631 / 0.784 | 0.7567 / 0.590 / 0.778 | 0.8377 / 0.617 / 0.776 | 0.7588 / 0.582 / 0.777 |
| aligned_ai | -0.1510 | 0.7478 / 0.996 / 0.748 | 0.6734 / 0.814 / 0.755 | 0.7512 / 0.998 / 0.752 | 0.6717 / 0.817 / 0.751 |
| aligned_everywhere_gbm | -0.1864 | 0.7129 / 0.964 / 0.714 | 0.6512 / 0.790 / 0.719 | 0.7200 / 0.980 / 0.725 | 0.6467 / 0.801 / 0.717 |

## world `bias_relu`

- admission_group_hidden: human acc privileged 0.899, underprivileged 0.499, privileged share 0.501, mean P(alignment) 0.487, slip 0.009
- admission_group_hidden_low_reliability: human acc privileged 0.752, underprivileged 0.503, privileged share 0.501, mean P(alignment) 0.459, slip 0.193
- admission_group_known: human acc privileged 0.898, underprivileged 0.509, privileged share 0.503, mean P(alignment) 0.488, slip 0.009
- admission_group_known_low_reliability: human acc privileged 0.751, underprivileged 0.505, privileged share 0.500, mean P(alignment) 0.457, slip 0.193

| method | raw gain | admission_group_hidden team / rel / ai | admission_group_hidden_low_reliability team / rel / ai | admission_group_known team / rel / ai | admission_group_known_low_reliability team / rel / ai |
|---|---|---|---|---|---|
| generator_bayes_policy | +0.0187 | 0.9268 / 0.957 / 0.935 | 0.8236 / 0.784 / 0.947 | 0.9397 / 0.982 / 0.952 | 0.8317 / 0.800 / 0.964 |
| proxy_gbm | +0.0017 | 0.9105 / 0.961 / 0.915 | 0.7981 / 0.785 / 0.913 | 0.9261 / 0.983 / 0.935 | 0.8159 / 0.800 / 0.943 |
| reference | +0.0000 | 0.9083 / 0.956 / 0.911 | 0.7988 / 0.773 / 0.909 | 0.9261 / 0.983 / 0.935 | 0.8159 / 0.800 / 0.943 |
| gbm_team | -0.0021 | 0.9121 / 0.943 / 0.920 | 0.8006 / 0.759 / 0.919 | 0.9273 / 0.975 / 0.941 | 0.8139 / 0.772 / 0.947 |
| gbm_specialists | -0.0024 | 0.9106 / 0.952 / 0.919 | 0.7964 / 0.765 / 0.909 | 0.9277 / 0.976 / 0.940 | 0.8112 / 0.778 / 0.933 |
| lr_team | -0.0053 | 0.9070 / 0.932 / 0.906 | 0.7918 / 0.745 / 0.888 | 0.9261 / 0.983 / 0.935 | 0.8159 / 0.800 / 0.943 |
| gbm_behavior_aware | -0.0057 | 0.9018 / 0.939 / 0.913 | 0.7985 / 0.758 / 0.914 | 0.9268 / 0.971 / 0.943 | 0.8171 / 0.782 / 0.951 |
| gbm_standard | -0.0147 | 0.8958 / 0.906 / 0.929 | 0.7987 / 0.769 / 0.930 | 0.9189 / 0.928 / 0.962 | 0.8181 / 0.780 / 0.964 |
| handrule_paper | -0.0383 | 0.8662 / 0.879 / 0.875 | 0.7711 / 0.747 / 0.873 | 0.9096 / 0.933 / 0.938 | 0.8061 / 0.782 / 0.938 |
| handrule_score_heuristic | -0.0484 | 0.8562 / 0.861 / 0.865 | 0.7650 / 0.739 / 0.866 | 0.9054 / 0.912 / 0.936 | 0.8044 / 0.770 / 0.938 |
| rrs | -0.0625 | 0.8041 / 0.851 / 0.780 | 0.7428 / 0.660 / 0.786 | — | — |
| behavior_aware_ai | -0.1213 | 0.8156 / 0.814 / 0.784 | 0.7391 / 0.675 / 0.786 | 0.8197 / 0.804 / 0.783 | 0.7400 / 0.665 / 0.785 |
| standard_ai | -0.1245 | 0.8132 / 0.813 / 0.786 | 0.7271 / 0.705 / 0.786 | 0.8169 / 0.806 / 0.785 | 0.7251 / 0.708 / 0.784 |
| complementary_ai | -0.1363 | 0.8430 / 0.627 / 0.784 | 0.7549 / 0.586 / 0.779 | 0.8425 / 0.619 / 0.776 | 0.7568 / 0.579 / 0.777 |
| aligned_ai | -0.1491 | 0.7515 / 0.983 / 0.757 | 0.6681 / 0.800 / 0.757 | 0.7569 / 0.983 / 0.762 | 0.6679 / 0.800 / 0.758 |
| aligned_everywhere_gbm | -0.1818 | 0.7180 / 0.957 / 0.724 | 0.6481 / 0.780 / 0.722 | 0.7259 / 0.972 / 0.734 | 0.6447 / 0.789 / 0.725 |

