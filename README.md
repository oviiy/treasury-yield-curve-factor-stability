# Treasury yield-curve factor stability

Research repository for a study of whether fixed factor loadings describe the U.S. Treasury yield curve. The primary analysis is the **full-sample** Su–Wang stability test. Monte Carlo experiments check its size and power at Treasury-like dimensions. Because Treasury residuals are strongly persistent, the current work adds a **serial-correlation-robust version of the test (sieve-AR GLS-J)**. Locating a break is a secondary, exploratory step.

> **Manuscript status:** [AI-written working draft](paper/working_draft.pdf), not peer reviewed. Its prose and presentation were prepared with AI assistance. Numerical results and interpretations require author and adviser verification. The original Su–Wang bootstrap over-rejects when errors persist across months; the new test addresses this in simulations but has no asymptotic theory yet, over-rejects when error volatility changes unevenly across maturities, and its empirical results are exploratory (not pre-registered).

## Start here

| Item | Content |
| --- | --- |
| [Research notebook](notebook/research_notebook.ipynb) | **Main working record.** Every step from the week-9 benchmark to the new test, with figures, tables and conclusions, in order. Reads committed results and runs short live examples. |
| [Main slides: full storyline](slides/Research_Update_Full_Storyline.pptx) | 24 slides: question, benchmark, why persistence breaks the test, the repairs tried, the new test, size/power, real data, open questions. |
| [Weekly quick take, 25 Nov 2026](slides/Weekly_Quick_Take_2026-11-25.pptx) | The current seven-page weekly update. |
| [Paper PDF](paper/working_draft.pdf) · [LaTeX](paper/working_draft.tex) | Full research argument; Section 7 is the serial-correlation-robust test. Explicitly marked as an AI-written draft. |
| [Key findings](results/key_findings.md) · [week-9 tables](results/final_tables.md) · [serial-correlation tables](results/serial_robust_tables.md) | Results in prose and frozen tables. |
| [Figure guide](figures/README.md) | Eleven figures, their data sources, and how to read them. |
| [Data](data/README.md) · [calibration](calibration/) | Public H.15 monthly yields, a labeled synthetic panel, and fitted calibration parameters. |
| [Reference](references/README.md) | Su and Wang (2017) and the literature used by the new test. |
| [Week 9 handover slides](slides/Week_9_handover.pptx) | The original 24-slide week-9 deck, kept for the record. |

## Five Python files

| File | Purpose |
| --- | --- |
| [`sw_test.py`](code/sw_test.py) | Local and global PCA, test statistic, paper-style independent-month covariance bootstrap, and early stopping. This is an equation-based implementation, not the authors' MATLAB program. |
| [`simulation.py`](code/simulation.py) | Calibrated three-factor null and three-to-four-factor DGPs, fixed seed blocks, full-sample experiment runner, size/power/dating metrics. |
| [`dating.py`](code/dating.py) | Sequential scan, rolling windows, split methods, and two experimental AR(1) bootstraps. |
| [`serial_robust.py`](code/serial_robust.py) | **New.** Sieve-AR GLS-J test (AIC-chosen AR filter, kernel-local vs full-sample loadings on filtered data, wild sieve bootstrap), the 33 serial-error DGP scenarios, a resumable Monte Carlo runner, and the Treasury empirical samples. |
| [`run.py`](code/run.py) | Command line entry point: benchmarks, synthetic panel, all eleven figures, and the new serial-error benchmark and empirical commands. |

`serial_robust.py` is a consolidation of the exploratory serial-correlation code. On 40 simulated panels (two calibrations, ten scenarios) it reproduces the exploratory code exactly: identical bootstrap p-values, 5% decisions, numbers of bootstrap draws and AR orders ([check](results/serial/port_equivalence.csv)). The week-9 refactor was checked the same way against the old weekly scripts.

## Reproduce

```bash
python -m pip install -r requirements.txt
python code/run.py figures
python code/run.py synthetic-data --dataset jkv --setup break --tau 240 --seed 100000
python code/run.py benchmark --dataset jkv --reps 10 --bootstrap 49 --jobs 1
python code/run.py serial-benchmark --scenarios null_ar06,alt_load_ar06 --reps 10 --bootstrap 199 --jobs 4
python code/run.py serial-empirical --bootstrap 999
jupyter nbconvert --to notebook --execute --inplace notebook/research_notebook.ipynb
```

The small benchmarks are smoke runs; they do **not** reproduce the frozen 250- and 500-panel tables. `serial-benchmark` uses the same seeds as the summarised runs, so `--reps 250` reproduces a cell of [`results/serial/final_method_size_power.csv`](results/serial/final_method_size_power.csv) at a computational cost. Benchmarks write resumable per-replication files in ignored `results/cells/`. `python code/run.py figures` uses committed tables and public input data only. The slides are rebuilt with `python slides/build_slides.py`.

## Interpretation and data boundary

Under independent simulated monthly errors, the final-seed full-sample 5% test has **3.8% size** on no-break panels and **100% power** against the calibrated full-strength fourth factor at three planted dates. Su and Wang's published power designs keep the factor count fixed and vary loadings; our three-to-four-factor power values are therefore not numerically comparable with theirs.

The JKV-based residuals have median lag-one autocorrelation about 0.63 (0.73 for the Treasury CMT panel). With AR(0.3) errors and no change, the original bootstrap rejects 62–95% of panels; the same happens in Su and Wang's own DGP 1 (2% → 88%). Narrowing the bandwidth reduces residual autocorrelation but raises false rejections, and AR(1) bootstraps restore size only by losing most power.

The **sieve-AR GLS-J** test keeps the Su–Wang null and local-versus-fixed comparison, filters each maturity's errors with an AIC-chosen AR model, and calibrates with a bootstrap that rebuilds the persistence. In 27 of 32 null cells (16 error designs × JKV/Treasury calibrations) its 5% size is 0.8–7.6%, and against a half-strength loading shift with AR(0.6) errors its power is 20.8% / 44.4% versus 7.5% / 23% for the AR bootstrap. When error volatility rises unevenly across maturities its size is 8–15%; this case is not separable from a loading change using yields alone, and is stated as an assumption. Exploratory real-data runs reject constant three-factor loadings over 1990–2009 (p = 0.026 JKV, 0.005 Treasury) but not four-factor loadings (0.196, 0.350).

The raw CRSP/Fama–Bliss yield panel is not redistributed; JKV empirical p-values are frozen results. Treasury results are rerun from the committed H.15 file. Source details and the H.15 sample boundary are in [data/README.md](data/README.md).
