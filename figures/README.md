# Figure guide

All eleven figures are rebuilt with `python code/run.py figures`. Monte Carlo figures read the frozen cells summarized in [`results/final_tables.md`](../results/final_tables.md); they are **not** estimates generated afresh by the plotting command.

| Figure | What to look for | Source |
| --- | --- | --- |
| [01 Treasury yields](01_treasury_yields.png) | Yield-curve shape and level change across years; real public CMT data. | H.15 snapshot, through August 2026. |
| [02 Simulation design](02_simulation_design.png) | Three continuing factors and one fourth factor switched on at a known month. | JKV calibration plus one synthetic panel. |
| [03 Full-sample size and power](03_fullsample_size_power.png) | False rejection without a break versus detection with a planted factor; this does not date the change. | Frozen Table G, final seeds. |
| [04 Bandwidth](04_bandwidth.png) | Narrower local bandwidth gains power but raises size; paper rule is the vertical dashed line. | Frozen Table C, tune seeds. |
| [05 Serial dependence](05_serial_dependence.png) | The paper bootstrap falsely rejects all no-break AR(1) panels once $\rho=0.3$; trial AR bootstraps trade size for power. | Frozen Table H, tune seeds. |
| [06 Dating](06_dating.png) | Detection and finding the month are different; the scan is usually late, while gated split dates strong middle/late changes better under the independent-error benchmark. | Frozen Tables B and G; not paired runs. |
| [07 Bandwidth and persistence](07_bandwidth_persistence.png) | Left: narrower h lowers real-data residual autocorrelation but never removes it. Right: narrower h makes the original test reject more often with AR(0.3) noise and no change. | `results/serial/bandwidth_real_grid.csv`, `bandwidth_original_test_summary.csv`. |
| [08 Su–Wang DGP with AR errors](08_suwang_dgp_serial.png) | In Su–Wang's own DGP 1, the paper's test and the authors' statistic go from about 2% to 88–100% size once errors are AR. | `results/serial/suwang_dgp_replication.csv`. |
| [09 New test: size](09_sieve_gls_size.png) | Sieve-AR GLS-J size across 16 error designs; the shaded band is the Monte Carlo interval around 5%. The two bottom rows (uneven volatility change) are the open limit. | `results/serial/simulation_all_runs_summary.csv` (`gls_aic_h2`). |
| [10 New test: power](10_sieve_gls_power.png) | Power grows with the size of the loading shift; at comparable size the new test beats the AR(1) bootstrap. Hatched bars have size above 10%. | `simulation_all_runs_summary.csv`, `ar_bootstrap_study_summary.csv`. |
| [11 Real-data p-values](11_empirical_pvalues.png) | Exploratory: three factors rejected, four not, in 1990–2009; 2010–2026 sits in the open-limit case. | `results/serial/empirical_frozen_pmax6.csv` (JKV), `empirical_treasury8_public.csv` (Treasury). |

The dating and AR(1) figures (05, 06) are diagnostic. Figures 07–10 are simulation evidence for the serial-correlation-robust test, which has no asymptotic theory yet; figure 11 is exploratory and was not pre-registered.
