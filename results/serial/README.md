# Serial-correlation results (summary tables)

Machine-readable summaries behind [`../serial_robust_tables.md`](../serial_robust_tables.md), figures 07–11, the notebook, the slides and Section 7 of the paper. Per-replication JSONL files are not committed; every summary row states its panel count.

| File | Content | Origin |
| --- | --- | --- |
| `residual_acf_summary.csv` | Lag-1/3/12 autocorrelation of real-data residuals by panel, fit and rank. | Week 10 diagnostics |
| `ar_bootstrap_study_summary.csv` | Original test vs AR(1) bootstraps (global, local, pooled, oracle) and prewhitening: size and power, T = 240, 200 panels. | Week 10 |
| `bandwidth_real_grid.csv` | Real-data residual autocorrelation and blocked-CV error across bandwidth multipliers, ranks 3–4. | Week 11 |
| `bandwidth_original_test_summary.csv` | Original test size/power at c = 0.5, 1, 2 under independent and AR(0.3/0.6) errors. | Week 11 |
| `bandwidth_selected_cv_summary.csv` | CV-selected bandwidth inside an AR(1) bootstrap. | Week 11 |
| `shape_test_key_results.csv` | Conditional Nelson–Siegel shape test and stress tests (identification limit). | Week 12 |
| `matlab_equivalence.csv` | Authors' MATLAB (Octave) vs our port on identical matrices. | Su–Wang audit |
| `suwang_dgp_replication.csv` | Su–Wang DGPs 1–4 with AR errors, five statistics. | Su–Wang audit |
| `simulation_all_runs_summary.csv` | Every serial-correlation Monte Carlo cell (runs `pilot`, `round2`–`round7`, `quick8`–`quick11`), all candidate methods. The final method is `gls_aic_h2`. | New test |
| `final_method_size_power.csv` | Final method, 25 headline cells (JKV / Treasury). | New test |
| `size_adjusted_power.csv` | Size-adjusted power (null 95% quantile as critical value) for the candidate methods. | New test |
| `port_equivalence.csv` | `code/serial_robust.py` vs the exploratory code on 40 panels: identical p-values, decisions, draws and AR orders. | This release |
| `empirical_frozen_pmax6.csv`, `empirical_frozen_pmax12.csv` | Real-data p-values (JKV and Treasury), B = 999, exploratory run. JKV uses the licensed panel. | New test |
| `empirical_treasury8_public.csv` | Treasury rerun from the committed H.15 file through August 2026 (`python code/run.py serial-empirical`). | This release |
| `bt_gls_size_power.csv`, `bt_gls_size_adjusted_power.csv` | BT-GLS J (Breitung–Tenhofen GLS factors + centred Su–Wang J) vs `gls_aic_h2`: rejection % (bootstrap and N(0,1)) and size-adjusted power; 250 panels in the key cells, 100–108 elsewhere. Columns `btJ_pca_*` use PC factors with the centred J. | Week 13 |
| `empirical_bt_frozen_jkv.csv`, `empirical_bt_treasury8_public.csv` | BT-GLS J on real data, B = 999, pmax 6 and 12 (JKV frozen; Treasury rerun with `python code/run.py serial-empirical --method bt`). | Week 13 |

Method names in `simulation_all_runs_summary.csv`: `gls_aic_h2` sieve-AR GLS-J (AIC order ≤ 6, h = 2h₀; final); `gls_M_wild` AR(1) filter, h₀; `gls_h05`/`gls_h2` AR(1) filter at 0.5h₀/2h₀; `gls_sieve`/`gls_sieve_h2` BIC order; `gls_ar2`, `gls_ar6_h2` fixed orders; `gls_bc` Kendall bias correction; `gls_iter` iterated factors; `gls_locvar_h2` local variance weights; `gls_split_h2` two-half weighted PCA; `gls_aicT_h2` T-growing maximum order; `gls_M_dwb4` dependent-wild innovations; `dwb_J_l6/12/24` original J with dependent wild bootstrap; `swqd_J_r`/`swqd_J_2r` quasi-differenced original Ĵ. Variants other than `gls_aic_h2` and the AR(1)/BIC/fixed-order filters are documented results, not implemented in `serial_robust.py`.
