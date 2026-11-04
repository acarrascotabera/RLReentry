# Final Monte Carlo

Set S6_mc_x0.1_a5; 95 % intervals: Clopper-Pearson (rates), bootstrap (percentiles).

| | fnpeg_hifi | fnpeg_ideal | rl_hifi | rl_ideal |
|---|---|---|---|---|
| n | 2000 | 2000 | 2000 | 2000 |
| reached handover | 100.0 % [99.8, 100.0] | 100.0 % [99.8, 100.0] | 100.0 % [99.8, 100.0] | 100.0 % [99.8, 100.0] |
| strictly feasible | 58.0 % [55.8, 60.2] | 69.9 % [67.8, 71.9] | 54.5 % [52.3, 56.7] | 65.7 % [63.6, 67.8] |
| feasible vs heat-rate floor | 67.8 % [65.7, 69.8] | 71.0 % [69.0, 73.0] | 57.2 % [55.0, 59.4] | 67.2 % [65.0, 69.2] |
| success rel_0p1pct | 0.0 % [0.0, 0.2] | 16.2 % [14.6, 17.8] | 0.0 % [0.0, 0.2] | 0.0 % [0.0, 0.2] |
| success rel_1pct | 1.4 % [0.9, 2.0] | 69.0 % [66.9, 71.0] | 0.0 % [0.0, 0.2] | 0.0 % [0.0, 0.2] |
| success vista_mc | 28.6 % [26.7, 30.7] | 69.2 % [67.1, 71.2] | 0.8 % [0.4, 1.2] | 7.5 % [6.4, 8.7] |
| success pos_1km | 18.3 % [16.6, 20.1] | 69.3 % [67.2, 71.3] | 0.9 % [0.5, 1.4] | 0.4 % [0.2, 0.8] |
| miss [km] p50 | 1.666 | 0.003 | 3.829 | 2.476 |
| miss [km] p95 | 6.357 | 0.022 | 10.490 | 6.023 |
| miss [km] p99.73 | 229.481 | 204.971 | 20.314 | 8.248 |
| miss [km] max | 307.024 | 287.416 | 123.037 | 58.262 |
| abs FPA error [deg] p50 | 1.006 | 0.022 | 4.267 | 1.170 |
| abs FPA error [deg] p95 | 10.085 | 0.069 | 12.490 | 9.445 |
| abs FPA error [deg] p99.73 | 19.314 | 18.231 | 16.073 | 12.663 |
| abs FPA error [deg] max | 20.387 | 18.756 | 18.287 | 15.505 |
| abs heading error [deg] p50 | 1.668 | 0.079 | 9.257 | 5.102 |
| abs heading error [deg] p95 | 7.840 | 0.183 | 20.591 | 12.300 |
| abs heading error [deg] p99.73 | 66.074 | 65.245 | 29.936 | 14.527 |
| abs heading error [deg] max | 68.695 | 67.401 | 30.808 | 23.735 |
| peak heat rate / limit p50 | 0.992 | 0.986 | 0.996 | 0.989 |
| peak heat rate / limit p95 | 1.054 | 1.034 | 1.060 | 1.038 |
| peak heat rate / limit p99.73 | 1.104 | 1.061 | 1.111 | 1.073 |
| peak heat rate / limit max | 1.115 | 1.094 | 1.120 | 1.094 |
| heat-rate excess over floor p50 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| heat-rate excess over floor p95 | 0.0034 | 0.0021 | 0.0114 | 0.0080 |
| heat-rate excess over floor p99.73 | 0.0516 | 0.0021 | 0.0312 | 0.0160 |
| heat-rate excess over floor max | 0.0687 | 0.0479 | 0.0552 | 0.0292 |
| peak load factor / limit p50 | 0.622 | 0.516 | 0.768 | 0.714 |
| peak load factor / limit p95 | 0.952 | 0.531 | 0.914 | 0.791 |
| peak load factor / limit p99.73 | 1.091 | 0.779 | 1.030 | 0.848 |
| peak load factor / limit max | 1.132 | 0.933 | 1.071 | 0.876 |
| peak dyn. pressure / limit p50 | 0.502 | 0.353 | 0.269 | 0.253 |
| peak dyn. pressure / limit p95 | 0.696 | 0.396 | 0.313 | 0.263 |
| peak dyn. pressure / limit p99.73 | 0.799 | 0.442 | 0.420 | 0.285 |
| peak dyn. pressure / limit max | 1.467 | 1.409 | 0.453 | 0.339 |
| heat load / SCvx optimum p50 | 1.109 | 1.109 | 1.210 | 1.207 |
| heat load / SCvx optimum p95 | 1.161 | 1.159 | 1.274 | 1.272 |
| heat load / SCvx optimum p99.73 | 1.197 | 1.194 | 1.324 | 1.323 |
| heat load / SCvx optimum max | 1.436 | 1.448 | 1.359 | 1.353 |
