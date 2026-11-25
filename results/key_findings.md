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

## Serial correlation and the new test (after week 9)

Simulation summaries are in [serial_robust_tables.md](serial_robust_tables.md) (Tables J–R); the full sequence is in the [research notebook](../notebook/research_notebook.ipynb).

| Exercise | Result | Interpretation |
| --- | --- | --- |
| Real residual persistence | Median lag-1 autocorrelation 0.63 (JKV) and 0.73 (Treasury CMT) with fixed three-factor loadings; positive at every maturity | The independent-month bootstrap assumption does not describe yield residuals. |
| AR(0.3) errors, no change, T = 240 | Original test size 95% (JKV) and 62.5% (Treasury); in Su–Wang's own DGP 1, 2% → 88% | Over-rejection is a property of the procedure under persistence, not of our data or code (our port matches the authors' MATLAB to 12 digits). |
| Narrower bandwidth (advisor suggestion) | Residual autocorrelation falls (JKV 0.63 → 0.22) but original-test size rises (62.5% → 98.5%) | Local loadings absorb persistent noise; tuning h alone cannot fix the test. |
| AR(1) bootstrap of the original statistic | Size 1–4.5%, but power against a half-strength loading shift with AR(0.6) errors only 7.5% / 23% | Fixes size by losing most power. |
| Conditional shape test (stable first 120 months) | Size 4–6%, power 58–90%; rejects 93–99% when only noise volatility rises 1.5× | A factor and an error common component are not separable without assumptions on the noise. |
| **Sieve-AR GLS-J** (AIC AR filter, local vs fixed loadings on filtered data, wild sieve bootstrap, h = 2h₀) | Size 0.8–7.6% in 27 of 32 null cells (AR up to 0.9, AR(2), ARMA(1,1), measurement noise, cross-maturity correlation changes, proportional volatility change); power 20.8% / 44.4% (half strength, AR(0.6)), 62.8% / 94.8% (full strength) | Keeps the Su–Wang null and comparison, holds size with persistent errors and has more power than an AR bootstrap. |
| Uneven volatility change across maturities | New-test size 14–15% (JKV), 8–10% (Treasury) | Open limit; stated as an assumption (stable relative error volatility within a window). |
| Real data, exploratory, B = 999 | 1990–2009: r = 3 rejected (p = 0.026 JKV, 0.005 Treasury); r = 4 not rejected (0.196, 0.350) | Part of the three-factor "loading change" may be a missing fourth factor. Not pre-registered; 2010–2026 not interpreted (zero lower bound). |

## Scope

- The project asks whether time-invariant factor loadings adequately describe Treasury yield curves. The primary method is the Su–Wang full-sample stability test; dating a change is secondary.
- The Su–Wang paper's power designs keep the factor count fixed and vary loadings. Our yield-curve calibration turns on a fourth factor after a known month. Rejection rates under these different alternatives are not directly comparable.
- JKV three-factor residuals have median lag-one autocorrelation around 0.63. The AR(1) simulations isolate one possible source of missized tests; they do **not** establish that empirical rejections are false. The sieve-AR GLS-J test is the current answer to this problem in simulations; it has no asymptotic theory yet.
- Raw CRSP / Fama–Bliss yields are not redistributed. The small `calibration/` files contain only fitted parameters needed for synthetic simulations.
