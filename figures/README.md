# Figure guide

All six figures are rebuilt with `python code/run.py figures`. Monte Carlo figures read the frozen cells summarized in [`results/final_tables.md`](../results/final_tables.md); they are **not** estimates generated afresh by the plotting command.

| Figure | What to look for | Source |
| --- | --- | --- |
| [01 Treasury yields](01_treasury_yields.png) | Yield-curve shape and level change across years; real public CMT data. | H.15 snapshot, through August 2026. |
| [02 Simulation design](02_simulation_design.png) | Three continuing factors and one fourth factor switched on at a known month. | JKV calibration plus one synthetic panel. |
| [03 Full-sample size and power](03_fullsample_size_power.png) | False rejection without a break versus detection with a planted factor; this does not date the change. | Frozen Table G, final seeds. |
| [04 Bandwidth](04_bandwidth.png) | Narrower local bandwidth gains power but raises size; paper rule is the vertical dashed line. | Frozen Table C, tune seeds. |
| [05 Serial dependence](05_serial_dependence.png) | The paper bootstrap falsely rejects all no-break AR(1) panels once $\rho=0.3$; trial AR bootstraps trade size for power. | Frozen Table H, tune seeds. |
| [06 Dating](06_dating.png) | Detection and finding the month are different; the scan is usually late, while gated split dates strong middle/late changes better under the independent-error benchmark. | Frozen Tables B and G; not paired runs. |

The dating and AR(1) figures are diagnostic. Neither establishes historical break dates or validates a corrected bootstrap for serially correlated Treasury residuals.
