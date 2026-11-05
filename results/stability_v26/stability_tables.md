# Stability analyses of v26 6M

## Margin sweeps (nominal + 100 cases at 0.1x)

within_margin: median landing error <= 2x baseline and feasible fraction >= baseline - 10 points.

| condition | group | magnitude | d_med | d_p90 | d_p95 | fpa_med | psi_med | feasible | reached | robust_score | bank_rate_rms | reversals | within_margin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| acc_2dps2 | acc | 2 | 1.66 | 3.09 | 3.81 | 2.11 | 3.33 | 0.762 | 1 | 0.708 | 1.09 | 4.08 | True |
| acc_5dps2 | acc | 5 | 1.74 | 3.27 | 4.04 | 1.64 | 3.7 | 0.762 | 1 | 0.674 | 1.02 | 4.08 | True |
| acc_10dps2 | acc | 10 | 1.72 | 3.19 | 3.95 | 1.75 | 3.53 | 0.762 | 1 | 0.671 | 1.01 | 4.08 | True |
| acc_20dps2 | acc | 20 | 1.68 | 3.15 | 3.89 | 1.84 | 3.29 | 0.762 | 1 | 0.672 | 1.03 | 4.08 | True |
| baseline | baseline | 0 | 1.79 | 3.31 | 4.05 | 1.55 | 3.94 | 0.762 | 1 | 0.668 | 0.983 | 4.08 | True |
| bias_-10deg | bias | -10 | 60.1 | 64 | 64.8 | 10.6 | 0.898 | 0.307 | 1 | 1.8 | 0.377 | 2.63 | False |
| bias_-5deg | bias | -5 | 23.7 | 27 | 27.5 | 2.88 | 6.59 | 0.703 | 1 | 1.4 | 0.449 | 3.33 | False |
| bias_-2deg | bias | -2 | 8.47 | 11 | 11.8 | 0.306 | 10.2 | 0.743 | 1 | 0.987 | 0.573 | 4.46 | False |
| bias_-1deg | bias | -1 | 4.44 | 6.94 | 7.71 | 0.145 | 9.32 | 0.743 | 1 | 0.766 | 0.662 | 4.05 | False |
| bias_-0.5deg | bias | -0.5 | 2.74 | 5.05 | 5.84 | 0.22 | 6.96 | 0.752 | 1 | 0.673 | 0.809 | 4.07 | True |
| bias_-0.25deg | bias | -0.25 | 2.02 | 4.18 | 4.93 | 0.673 | 5.1 | 0.752 | 1 | 0.658 | 0.892 | 4.05 | True |
| bias_+0.25deg | bias | 0.25 | 1.65 | 2.83 | 3.23 | 2.82 | 2.85 | 0.762 | 1 | 0.705 | 1.08 | 4.06 | True |
| bias_+0.5deg | bias | 0.5 | 1.71 | 2.74 | 3.28 | 4.29 | 2.95 | 0.762 | 1 | 0.759 | 1.19 | 4.06 | True |
| bias_+1deg | bias | 1 | 2.15 | 3.48 | 3.8 | 6.58 | 3.94 | 0.772 | 1 | 0.874 | 1.39 | 4.06 | True |
| bias_+2deg | bias | 2 | 3.6 | 4.98 | 5.26 | 9.33 | 4.05 | 0.752 | 1 | 1.03 | 1.81 | 4.03 | False |
| bias_+5deg | bias | 5 | 5.97 | 6.66 | 6.92 | 13.5 | 30.1 | 0.703 | 1 | 1.22 | 0.896 | 3.11 | False |
| bias_+10deg | bias | 10 | 9.85 | 11.5 | 11.8 | 12.5 | 33.1 | 0.406 | 1 | 1.28 | 0.677 | 3 | False |
| delay_0.5s | delay | 0.5 | 2.1 | 3.89 | 4.69 | 0.69 | 6.29 | 0.762 | 1 | 0.67 | 0.91 | 4.08 | True |
| delay_1s | delay | 1 | 2.42 | 4.46 | 5.3 | 0.48 | 8.39 | 0.762 | 1 | 0.683 | 0.863 | 4.07 | True |
| delay_2s | delay | 2 | 3.66 | 5.97 | 6.82 | 0.218 | 12.9 | 0.762 | 1 | 0.764 | 0.824 | 4.08 | False |
| delay_4s | delay | 4 | 6.4 | 8.3 | 9.09 | 1.82 | 17.7 | 0.762 | 1 | 0.938 | 0.996 | 5.02 | False |
| delay_8s | delay | 8 | 10.4 | 13.2 | 13.3 | 2.49 | 25.8 | 0 | 1 | 1.16 | 0.771 | 3.49 | False |
| gain_0.5 | gain | 0.5 | 565 | 662 | 671 | 12.7 | 54.5 | 0.772 | 1 | 2.75 | 0.207 | 2.06 | False |
| gain_0.7 | gain | 0.7 | 175 | 203 | 209 | 12.2 | 57.5 | 0.772 | 1 | 2.25 | 0.353 | 2.06 | False |
| gain_0.85 | gain | 0.85 | 80 | 89.6 | 93.6 | 10.4 | 9.91 | 0.772 | 1 | 1.91 | 0.367 | 3.06 | False |
| gain_0.95 | gain | 0.95 | 13.8 | 17.2 | 17.9 | 0.593 | 11 | 0.762 | 1 | 1.18 | 0.498 | 3.28 | False |
| gain_0.97 | gain | 0.97 | 7.64 | 10.5 | 11.6 | 0.147 | 12.1 | 0.762 | 1 | 0.96 | 0.561 | 4.59 | False |
| gain_1.03 | gain | 1.03 | 3.18 | 4.63 | 4.9 | 10.1 | 10.4 | 0.752 | 1 | 1.06 | 1.4 | 4.1 | True |
| gain_1.05 | gain | 1.05 | 4.7 | 6.09 | 6.32 | 13.1 | 14.3 | 0.752 | 1 | 1.18 | 1.57 | 4.1 | False |
| gain_1.15 | gain | 1.15 | 9.62 | 10.9 | 11.1 | 19.1 | 23.1 | 0.743 | 1 | 1.36 | 2.03 | 3.96 | False |
| gain_1.3 | gain | 1.3 | 11.8 | 18.5 | 20 | 18.9 | 43 | 0.0891 | 1 | 1.63 | 1.24 | 3.28 | False |
| gain_1.5 | gain | 1.5 | 28.2 | 33.7 | 34.4 | 15.7 | 36.4 | 0 | 1 | 1.96 | 1.56 | 3.14 | False |
| hifi_stack | hifi | 1 | 3.68 | 8.16 | 8.82 | 3.6 | 9.69 | 0.644 | 1 | 0.917 | 1.28 | 4.38 | False |
| imp_V-100_late | impulse_V-100 | 0.08 | 10.4 | 13.1 | 14 | 0.424 | 10.6 | 0.762 | 1 | 1.06 | 0.54 | 4.73 | False |
| imp_V-100_mid | impulse_V-100 | 0.35 | 1.66 | 2.82 | 3.36 | 1.82 | 3.31 | 0.762 | 1 | 0.664 | 1.03 | 4.08 | True |
| imp_V-100_early | impulse_V-100 | 0.7 | 1.63 | 2.94 | 3.92 | 2.13 | 3.57 | 0.762 | 1 | 0.681 | 1.05 | 4.08 | True |
| imp_fpa+1_late | impulse_fpa+1 | 0.08 | 2.51 | 4.74 | 5.57 | 0.224 | 8.04 | 0.762 | 1 | 0.664 | 0.828 | 4.07 | True |
| imp_fpa+1_mid | impulse_fpa+1 | 0.35 | 1.75 | 3.06 | 3.96 | 2.4 | 3.72 | 0.762 | 1 | 0.696 | 1.06 | 4.07 | True |
| imp_fpa+1_early | impulse_fpa+1 | 0.7 | 1.85 | 3.98 | 5.06 | 1.39 | 3.96 | 0.762 | 1 | 0.672 | 0.958 | 4.07 | True |
| imp_fpa-1_late | impulse_fpa-1 | 0.08 | 1.86 | 3.25 | 3.8 | 4.68 | 4.79 | 0.762 | 1 | 0.784 | 1.25 | 4.08 | True |
| imp_fpa-1_mid | impulse_fpa-1 | 0.35 | 1.82 | 2.64 | 2.89 | 4.88 | 3.79 | 0.762 | 1 | 0.778 | 1.27 | 4.08 | True |
| imp_fpa-1_early | impulse_fpa-1 | 0.7 | 1.69 | 2.7 | 2.91 | 2.74 | 3.25 | 0.624 | 1 | 0.706 | 1.09 | 4.08 | False |
| imp_psi+5_late | impulse_psi+5 | 0.08 | 8.5 | 9.3 | 9.65 | 15.7 | 33.7 | 0.752 | 1 | 1.31 | 0.774 | 3.1 | False |
| imp_psi+5_mid | impulse_psi+5 | 0.35 | 6.03 | 7.93 | 8.42 | 0.175 | 12.2 | 0.762 | 1 | 0.891 | 0.82 | 4.18 | False |
| imp_psi+5_early | impulse_psi+5 | 0.7 | 1.94 | 4.31 | 4.81 | 0.39 | 6.79 | 0.762 | 1 | 0.642 | 0.954 | 4.06 | True |
| lag_0.5s | lag | 0.5 | 1.94 | 3.64 | 4.43 | 1 | 4.86 | 0.762 | 1 | 0.666 | 0.913 | 4.08 | True |
| lag_1s | lag | 1 | 2.23 | 4.23 | 5.09 | 0.488 | 7.68 | 0.762 | 1 | 0.675 | 0.837 | 4.07 | True |
| lag_2s | lag | 2 | 3.27 | 5.52 | 6.36 | 0.22 | 11.9 | 0.762 | 1 | 0.734 | 0.696 | 4.07 | True |
| lag_4s | lag | 4 | 5.16 | 7.29 | 7.69 | 0.299 | 17.2 | 0.762 | 1 | 0.858 | 0.543 | 4.02 | False |
| lag_8s | lag | 8 | 6.25 | 8.31 | 8.77 | 2.14 | 26.7 | 0 | 1 | 1.06 | 0.481 | 3.54 | False |
| nav_x0.5 | nav | 0.5 | 2.11 | 4.4 | 4.74 | 1.75 | 5.78 | 0.762 | 1 | 0.716 | 1.57 | 5.86 | True |
| nav_x1 | nav | 1 | 3.35 | 7.51 | 8.72 | 2.03 | 9.23 | 0.752 | 1 | 0.813 | 2.24 | 9.33 | True |
| nav_x2 | nav | 2 | 6.14 | 17.4 | 20.4 | 2.48 | 9.62 | 0.653 | 1 | 0.984 | 3.17 | 14.7 | False |
| nav_x4 | nav | 4 | 13.2 | 46.7 | 51.7 | 3.36 | 8.11 | 0.564 | 1 | 1.29 | 4.5 | 24.9 | False |
| nav_x8 | nav | 8 | 35.6 | 117 | 143 | 5.8 | 10 | 0.356 | 1 | 1.66 | 7.7 | 45.6 | False |
| navgm_x1 | nav_gm | 1 | 4.21 | 12.1 | 13.7 | 3.31 | 9.57 | 0.743 | 1 | 0.916 | 1.03 | 4.26 | False |
| navgm_x2 | nav_gm | 2 | 7.72 | 24.8 | 34.3 | 5.58 | 9.23 | 0.693 | 1 | 1.13 | 0.985 | 4.38 | False |
| navgm_x4 | nav_gm | 4 | 17.1 | 58.4 | 68.7 | 9.79 | 9.46 | 0.426 | 1 | 1.41 | 0.923 | 4.98 | False |
| rate_5dps | rate | 5 | 1.85 | 3.29 | 4.03 | 0.821 | 4.43 | 0.762 | 1 | 0.633 | 0.863 | 4.08 | True |
| rate_10dps | rate | 10 | 1.8 | 3.31 | 4.05 | 1.55 | 3.94 | 0.762 | 1 | 0.666 | 0.97 | 4.08 | True |
| rate_20dps | rate | 20 | 1.79 | 3.31 | 4.05 | 1.55 | 3.94 | 0.762 | 1 | 0.668 | 0.982 | 4.08 | True |
| rate_30dps | rate | 30 | 1.79 | 3.31 | 4.05 | 1.55 | 3.94 | 0.762 | 1 | 0.668 | 0.983 | 4.08 | True |
| rhobump_-0.3 | rho_bump | -0.3 | 3.01 | 3.78 | 4.02 | 7.17 | 6.96 | 0.743 | 1 | 0.933 | 1.55 | 4.08 | True |
| rhobump_+0.3 | rho_bump | 0.3 | 9.36 | 12.2 | 13 | 0.125 | 11.4 | 0.762 | 1 | 1.02 | 0.565 | 4.32 | False |
| rhogm_x0.5 | rho_gm | 0.5 | 1.83 | 3.54 | 4 | 2.1 | 3.63 | 0.752 | 1 | 0.695 | 1.02 | 4.14 | True |
| rhogm_x1 | rho_gm | 1 | 1.88 | 3.59 | 4.61 | 2.56 | 4.11 | 0.644 | 1 | 0.753 | 1.08 | 4.14 | False |
| rhogm_x2 | rho_gm | 2 | 2.28 | 5.73 | 6.42 | 3.43 | 5.8 | 0.426 | 1 | 0.895 | 1.19 | 4.14 | False |
| rhowave_0.1 | rho_wave | 0.1 | 1.75 | 4.7 | 5.95 | 1.06 | 6.05 | 0.535 | 1 | 0.772 | 1.01 | 4.06 | False |
| rhowave_0.2 | rho_wave | 0.2 | 4.24 | 9.02 | 10.2 | 0.424 | 8.27 | 0 | 1 | 1.23 | 0.781 | 4.13 | False |
| rhowave_0.3 | rho_wave | 0.3 | 7.49 | 12.4 | 12.9 | 1.31 | 4.03 | 0 | 1 | 1.61 | 0.702 | 4.29 | False |
| wind_x0.5 | wind | 0.5 | 1.83 | 3.11 | 3.39 | 3.04 | 3.58 | 0.762 | 1 | 0.723 | 1.11 | 4.08 | True |
| wind_x1 | wind | 1 | 2.19 | 3.55 | 4.12 | 4.27 | 4.26 | 0.782 | 1 | 0.794 | 1.24 | 4.08 | True |
| wind_x2 | wind | 2 | 2.93 | 5.36 | 5.9 | 6.03 | 6.68 | 0.713 | 1 | 0.905 | 1.48 | 4.09 | True |

## Terminal sensitivity, closed vs open loop (nominal mission)

Mean terminal displacement [km] over the + and - perturbation.

| param | absm | closed | open | attenuation |
|---|---|---|---|---|
| ic_V_mps | 1 | 0.65 | 82.7 | 127 |
| ic_V_mps | 5 | 3.48 | 414 | 119 |
| ic_alt_m | 1 | 0.938 | 26.7 | 28.4 |
| ic_alt_m | 5 | 3.94 | 137 | 34.8 |
| ic_fpa_deg | 1 | 0.549 | 52.1 | 94.9 |
| ic_fpa_deg | 5 | 2.47 | 256 | 103 |
| ic_lat_deg | 1 | 0.0309 | 11.1 | 360 |
| ic_lat_deg | 5 | 0.177 | 57.8 | 327 |
| ic_psi_deg | 1 | 0.0914 | 25.8 | 282 |
| ic_psi_deg | 5 | 0.51 | 138 | 270 |
| k_CD | 1 | 1.06 | 139 | 131 |
| k_CD | 5 | 4.36 | 683 | 157 |
| k_CL | 1 | 0.814 | 137 | 169 |
| k_CL | 5 | 3.23 | 652 | 202 |
| k_mass | 1 | 0.272 | 4.22 | 15.5 |
| k_mass | 5 | 1.34 | 15.6 | 11.6 |
| k_rho | 1 | 0.0505 | 3.43 | 67.9 |
| k_rho | 5 | 0.256 | 15.7 | 61.5 |

## Recovery after impulsive kicks (nominal mission)

| mode | kick | phase | d_km | dfpa_deg | dpsi_deg | max_ratio | feasible |
|---|---|---|---|---|---|---|---|
| closed | none | none | 1.62 | -1.24 | -3.55 | 0.986 | True |
| closed | fpa-1 | early | 1.23 | -2.65 | -0.82 | 0.995 | True |
| closed | fpa+1 | early | 1.72 | -1.39 | -2.81 | 0.986 | True |
| closed | psi+5 | early | 1.98 | -0.406 | -7 | 0.986 | True |
| closed | V-100 | early | 1.28 | -1.95 | -2.2 | 0.986 | True |
| closed | fpa-1 | mid | 1.68 | -4.65 | 2.93 | 0.986 | True |
| closed | fpa+1 | mid | 1.41 | -2.48 | -0.0221 | 0.986 | True |
| closed | psi+5 | mid | 6.16 | -0.107 | -12.2 | 0.986 | True |
| closed | V-100 | mid | 1.36 | -1.83 | -2.08 | 0.986 | True |
| closed | fpa-1 | late | 1.58 | -4.47 | 3.12 | 0.986 | True |
| closed | fpa+1 | late | 2.81 | -0.138 | -8.43 | 0.986 | True |
| closed | psi+5 | late | 8.24 | -15.8 | -34.1 | 0.986 | True |
| closed | V-100 | late | 10.5 | -0.484 | -10.8 | 0.986 | True |
| open | none | none | 1.62 | -1.24 | -3.55 | 0.986 | True |
| open | fpa-1 | early | 147 | -2.5 | -6.48 | 1.06 | False |
| open | fpa+1 | early | 155 | -15.2 | 29.6 | 0.986 | True |
| open | psi+5 | early | 262 | -3.65 | 14.9 | 0.986 | True |
| open | V-100 | early | 160 | -2.93 | -1.93 | 0.986 | True |
| open | fpa-1 | mid | 72.3 | -1.79 | -9.53 | 0.986 | True |
| open | fpa+1 | mid | 71.4 | -11 | 20.7 | 0.986 | True |
| open | psi+5 | mid | 79.6 | -1.15 | 2.06 | 0.986 | True |
| open | V-100 | mid | 59.7 | -1.64 | -11.6 | 0.986 | True |
| open | fpa-1 | late | 13.8 | -0.15 | -12.7 | 0.986 | True |
| open | fpa+1 | late | 11.4 | -3.84 | 5.22 | 0.986 | True |
| open | psi+5 | late | 21.6 | -1.19 | 0.99 | 0.986 | True |
| open | V-100 | late | 26.2 | -1.36 | -14.5 | 0.986 | True |

## corridor

121 grid points: reached and strictly feasible 4 %, heat-rate floor > 1 (infeasible for any guidance) 70 %, guidance-attributable feasible 4 %, landing error <= 5 km 31 % (and attributable feasible 2 %), median landing error 35.17 km.

## footprint

169 grid points: reached and strictly feasible 93 %, heat-rate floor > 1 (infeasible for any guidance) 0 %, guidance-attributable feasible 93 %, landing error <= 5 km 82 % (and attributable feasible 76 %), median landing error 2.18 km.

## Policy Jacobian norm

Over 17004 decisions on the nominal + 20 cases: median 66.7, p95 88.8, max 91.7 deg per unit normalized feature.

## Combined realistic perturbations (300 cases per set)

nav errors at spec + 1 s delay + 1 s lag + 30 deg/s bank rate + 10 % density waves.

| set | condition | d_med | d_p95 | fpa_med | psi_med | feasible | robust_score |
|---|---|---|---|---|---|---|---|
| S4_scale_x0.1 | baseline | 1.69 | 3.7 | 1.54 | 4.22 | 0.803 | 0.659 |
| S4_scale_x0.1 | combined | 4.7 | 13.7 | 1.29 | 11 | 0.317 | 0.945 |
| S4_scale_x0.25 | baseline | 2.7 | 7.38 | 2.12 | 7.82 | 0.597 | 0.885 |
| S4_scale_x0.25 | combined | 4.89 | 15.6 | 1.64 | 11 | 0.337 | 1.02 |
