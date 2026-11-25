# Data and provenance

| File | What it contains | Status |
| --- | --- | --- |
| `treasury_cmt_h15_monthly_1962_2026.csv` | Monthly H.15 Treasury constant-maturity par yields, maturities named in months (`m1` through `m360`); 777 rows, January 1962 to a partial September 2026 observation. | Public Federal Reserve Board series. Missing cells reflect maturity availability; do not replace them with zero. |
| `synthetic_jkv_example.csv` | 480 months by 17 maturities, generated from the public JKV calibration with seed 100000, a fourth factor entering at month 240, independent Gaussian errors, and full factor strength. | **Synthetic**, not historical Treasury observations. Values are on the demeaned yield scale. |
| `../calibration/model_jkv.npz` | Three- and four-factor fitted loadings, factor AR(1) coefficients, innovation scales and residual scales. | Derived calibration parameters, not the original CRSP records. |
| `../calibration/model_treasury_cmt.npz` | Equivalent fitted parameters for the eight-maturity CMT panel. | Derived calibration parameters. |
| `../calibration/model_treasury8_1990_2009.npz` | Nelson–Siegel/Svensson parameters fitted to the eight-maturity Treasury par curve, January 1990–December 2009; used by the serial-correlation experiments (`code/serial_robust.py`). | Derived from public data. In this window the month-end Treasury par curve equals the committed H.15 CMT values exactly. |

The manuscript's CMT analysis ends in **August 2026**. The CSV retains the partial September 2026 row so its provenance remains clear; the plot command excludes that row. To recreate `synthetic_jkv_example.csv`, run:

```bash
python code/run.py synthetic-data --dataset jkv --setup break --tau 240 --seed 100000
```

The raw JKV/Fama–Bliss panel used for calibration is based on CRSP Treasury data and is not included for redistribution. Reproducing its *empirical* tests or refitting the calibration requires the appropriately licensed source panel. The public simulation commands use only the committed `.npz` calibration files. The JKV empirical p-values of the serial-correlation-robust test (`results/serial/empirical_frozen_*.csv`) are frozen results from the licensed panel; the Treasury empirical results are rerun from the committed H.15 file (`python code/run.py serial-empirical`).

Sources: [Federal Reserve H.15 release](https://www.federalreserve.gov/releases/h15/) and [Federal Reserve Board copyright/data notice](https://www.federalreserve.gov/disclaimer.htm). The CMT CSV is a project snapshot, not a live data feed.
