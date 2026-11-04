# Final Monte Carlo

Set S6_mc_x0.1_a5; 95 % intervals: Clopper-Pearson (rates), bootstrap (percentiles).

| | fnpeg_hifi | fnpeg_ideal | rl_hifi | rl_ideal |
|---|---|---|---|---|
| n | 2000 | 2000 | 2000 | 2000 |
| reached handover | 100.0 % [99.8, 100.0] | 100.0 % [99.8, 100.0] | 100.0 % [99.8, 100.0] | 100.0 % [99.8, 100.0] |
| strictly feasible | 58.0 % [55.8, 60.1] | 69.9 % [67.8, 71.9] | 54.5 % [52.3, 56.7] | 65.7 % [63.6, 67.8] |
| feasible vs heat-rate floor | 67.5 % [65.3, 69.5] | 71.0 % [69.0, 73.0] | 57.2 % [55.0, 59.4] | 67.2 % [65.0, 69.2] |
| success rel_0p1pct | 0.1 % [0.0, 0.3] | 17.4 % [15.8, 19.1] | 0.0 % [0.0, 0.2] | 0.0 % [0.0, 0.2] |
| success rel_1pct | 1.2 % [0.8, 1.8] | 69.8 % [67.8, 71.9] | 0.0 % [0.0, 0.2] | 0.0 % [0.0, 0.2] |
| success vista_mc | 28.1 % [26.2, 30.2] | 69.9 % [67.8, 71.9] | 0.8 % [0.4, 1.2] | 7.5 % [6.4, 8.7] |
| success pos_1km | 18.0 % [16.3, 19.8] | 69.9 % [67.8, 71.9] | 0.9 % [0.5, 1.4] | 0.4 % [0.2, 0.8] |
| miss [km] p50 | 1.592 | 0.002 | 3.829 | 2.476 |
| miss [km] p95 | 4.680 | 0.006 | 10.490 | 6.023 |
| miss [km] p99.73 | 8.477 | 0.027 | 20.314 | 8.248 |
| miss [km] max | 34.108 | 0.080 | 123.037 | 58.262 |
| abs FPA error [deg] p50 | 0.969 | 0.021 | 4.267 | 1.170 |
| abs FPA error [deg] p95 | 3.464 | 0.059 | 12.490 | 9.445 |
| abs FPA error [deg] p99.73 | 14.018 | 0.104 | 16.073 | 12.663 |
| abs FPA error [deg] max | 17.822 | 0.393 | 18.287 | 15.505 |
| abs heading error [deg] p50 | 1.628 | 0.073 | 9.257 | 5.102 |
| abs heading error [deg] p95 | 5.262 | 0.156 | 20.591 | 12.300 |
| abs heading error [deg] p99.73 | 9.589 | 0.192 | 29.936 | 14.527 |
| abs heading error [deg] max | 11.863 | 0.201 | 30.808 | 23.735 |
| peak heat rate / limit p50 | 0.992 | 0.986 | 0.996 | 0.989 |
| peak heat rate / limit p95 | 1.054 | 1.034 | 1.060 | 1.038 |
| peak heat rate / limit p99.73 | 1.105 | 1.067 | 1.111 | 1.073 |
| peak heat rate / limit max | 1.261 | 1.205 | 1.120 | 1.094 |
| heat-rate excess over floor p50 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| heat-rate excess over floor p95 | 0.0034 | 0.0021 | 0.0114 | 0.0080 |
| heat-rate excess over floor p99.73 | 0.0519 | 0.0202 | 0.0312 | 0.0160 |
| heat-rate excess over floor max | 0.2148 | 0.1399 | 0.0552 | 0.0292 |
| peak load factor / limit p50 | 0.621 | 0.516 | 0.768 | 0.714 |
| peak load factor / limit p95 | 0.967 | 0.526 | 0.914 | 0.791 |
| peak load factor / limit p99.73 | 1.089 | 0.550 | 1.030 | 0.848 |
| peak load factor / limit max | 1.146 | 0.589 | 1.071 | 0.876 |
| peak dyn. pressure / limit p50 | 0.508 | 0.355 | 0.269 | 0.253 |
| peak dyn. pressure / limit p95 | 0.700 | 0.407 | 0.313 | 0.263 |
| peak dyn. pressure / limit p99.73 | 0.843 | 0.568 | 0.420 | 0.285 |
| peak dyn. pressure / limit max | 1.275 | 1.093 | 0.453 | 0.339 |
| heat load / SCvx optimum p50 | 1.110 | 1.111 | 1.210 | 1.207 |
| heat load / SCvx optimum p95 | 1.167 | 1.166 | 1.274 | 1.272 |
| heat load / SCvx optimum p99.73 | 1.259 | 1.263 | 1.324 | 1.323 |
| heat load / SCvx optimum max | 1.454 | 1.454 | 1.359 | 1.353 |
