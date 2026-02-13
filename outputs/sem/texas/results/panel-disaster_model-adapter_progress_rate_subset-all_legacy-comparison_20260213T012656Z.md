# SEM Comparison: Texas Pipeline vs Legacy

| Model | texas::adapter_progress_rate [disaster|all] | legacy.model_comparison::Baseline | legacy.model_comparison::M1_RemoveRatioOblig | legacy.model_comparison::M2_RemoveTimeliness | legacy.model_comparison::M3_RemoveBoth | legacy.model_comparison::M4_DurationInCapacity | legacy.model_comparison::M5_SingleFactor | legacy.model_comparison::M6_RatiosOnly | legacy.model_comparison::M7_RemoveQBQVar | legacy.model_comparison::M8_Minimal4Indicators | legacy.model_comparison::M9_DirectEffects | legacy.fit_stats::all_last_observed | legacy.fit_stats::exp_optimal_v1_censored | legacy.fit_stats::exp_optimal_v1_last_observed | legacy.fit_stats::progress_rate_short_series_last_observed | legacy.fit_stats::time_to_50pct_last_observed | legacy.fit_stats::with_all_covariates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| DoF | 0.0000 | 8.0000 | 4.0000 | 4.0000 | 1.0000 | 4.0000 | 5.0000 | 1.0000 | 4.0000 | 1.0000 | 2.0000 | 8.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 27.0000 |
| chi2 | 147.8545 | 94.1479 | 298.7388 | 78.1745 | 30.6497 | 79.1779 | 79.1810 | 61.7443 | 50.2367 | 30.2669 | 31.6405 | 93.1390 | 0.3804 | 0.1919 | 5.3240 | 21.9642 | 462.6740 |
| p_value |  | 1.1102e-16 | 0.0000 | 4.4409e-16 | 3.0908e-08 | 2.2204e-16 | 1.2212e-15 | 3.8858e-15 | 3.2225e-10 | 3.7650e-08 | 1.3470e-07 | 1.1102e-16 | 0.5374 | 0.6613 | 0.0210 | 2.7779e-06 | 0.0000 |
| CFI | -0.0207 | 0.7810 | -1.5836 | 0.7345 | 0.6348 | 0.7309 | 0.7345 | 0.7420 | 0.8292 | 0.8756 | 0.8741 | 0.8252 | 1.3793 | 1.0090 | 0.9341 | 0.7944 | -0.6274 |
| TLI |  | 0.5893 | -5.4589 | 0.3363 | -1.1914 | 0.3273 | 0.4690 | -0.5482 | 0.5731 | 0.2539 | 0.6222 | 0.6723 | 3.2757 | 1.0539 | 0.6045 | -0.2334 | -1.3507 |
| RMSEA | inf | 0.4102 | 1.0730 | 0.5383 | 0.6806 | 0.5419 | 0.4815 | 0.9742 | 0.4250 | 0.6762 | 0.4812 | 0.4078 | 0.0000 | 0.0000 | 0.2385 | 0.5723 | 0.8034 |
| AIC | 10.7470 | 23.1031 | 12.8080 | 19.5946 | 17.0569 | 19.5638 | 17.5637 | 16.1002 | 20.4543 | 17.0687 | 15.0264 | 23.1342 | 17.9718 | 17.9941 | 17.8617 | 17.3242 | 0.4097 |
| BIC | 31.5300 | 51.3702 | 36.7263 | 43.5129 | 36.6264 | 43.4820 | 39.3075 | 35.6697 | 44.3725 | 36.6382 | 32.4215 | 51.4012 | 29.6344 | 37.5636 | 38.9560 | 36.8937 | 23.0554 |
