# Current findings (working draft)

These are simulation results, not evidence that a particular historical break occurred. Detailed cells and Monte Carlo standard errors are in [final_tables.md](final_tables.md).

| Exercise | Result | Interpretation |
| --- | --- | --- |
| JKV-calibrated, three-factor null, serially independent errors; 500 final-seed panels | Full-sample size at nominal 1% / 5% / 10%: 1.2% / 3.8% / 7.6% | Acceptable, somewhat conservative size under this error design. |
| Fourth factor enters at months 160 / 240 / 320; baseline strength | Full-sample power at 5%: 100% at all three positions | Strong factor-structure changes are easy to detect in this design. |
| Fourth-factor strength halved; same positions | Full-sample power at 5%: 91.6% / 91.4% / 82.4% | A later or weaker change is harder to detect. |
| No break; AR(1) errors with coefficient 0.3 or higher | Original Su–Wang bootstrap size at nominal 5%: 100% in the tested cells | Serial persistence can produce rejection without a planted change. |
| Alternative AR(1) bootstraps | Global-residual version controls size but loses power; local-residual version gains power but over-rejects | Neither is adopted as a correction. |
| Sequential scan, serially independent errors | Procedure-level size 7%–13% at nominal 1%; strong-break date roughly 20 months late | Repeated windows and the first-rejection dating rule have separate costs. |
| Full-sample test followed by least-squares split | 5% size 3.8% on final seeds; strong-break dates within ±24 months in 57% / 80% / 87% | Promising for dating middle or late strong changes, conditional on a valid first-stage test. |

## Scope

- The project asks whether time-invariant factor loadings adequately describe Treasury yield curves. The primary method is the Su–Wang full-sample stability test; dating a change is secondary.
- The Su–Wang paper's power designs keep the factor count fixed and vary loadings. Our yield-curve calibration turns on a fourth factor after a known month. Rejection rates under these different alternatives are not directly comparable.
- JKV three-factor residuals have median lag-one autocorrelation around 0.63. The AR(1) simulations isolate one possible source of missized tests; they do **not** establish that empirical rejections are false.
- Raw CRSP / Fama–Bliss yields are not redistributed. The small `calibration/` files contain only fitted parameters needed for synthetic simulations.
