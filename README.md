# Treasury yield-curve factor stability

Research repository for a study of whether fixed factor loadings describe the U.S. Treasury yield curve. The primary analysis is the **full-sample** Su–Wang stability test. Monte Carlo experiments check its size and power at Treasury-like dimensions; locating a break is a secondary, exploratory step.

> **Manuscript status:** [AI-written working draft](paper/working_draft.pdf), not peer reviewed. Its prose and presentation were prepared with AI assistance. Numerical results and interpretations require author and adviser verification. The central unresolved issue is serial correlation in Treasury residuals: the original bootstrap may over-reject when errors persist across months. Historical rejections and estimated break dates are provisional.

## Start here

| Item | Content |
| --- | --- |
| [Paper PDF](paper/working_draft.pdf) · [LaTeX](paper/working_draft.tex) | Full research argument, methods, limitations, and figures; explicitly marked as an AI-written draft. |
| [Key findings](results/key_findings.md) · [frozen result tables](results/final_tables.md) | Results from the completed week-nine simulation runs. |
| [Figure guide](figures/README.md) | Six figures, their data sources, and how to read them. |
| [Data](data/README.md) · [calibration](calibration/) | Public H.15 monthly yields, a labeled synthetic panel, and fitted parameters for the JKV-based simulation. |
| [Reference](references/README.md) | Citation and official access links for Su and Wang (2017), including the licensed 2015 working-paper version. |

## Four Python files

| File | Purpose |
| --- | --- |
| [`sw_test.py`](code/sw_test.py) | Local and global PCA, test statistic, paper-style independent-month covariance bootstrap, and early stopping. This is an equation-based implementation, not the authors' MATLAB program. |
| [`simulation.py`](code/simulation.py) | Calibrated three-factor null and three-to-four-factor DGPs, fixed seed blocks, full-sample experiment runner, size/power/dating metrics. |
| [`dating.py`](code/dating.py) | Sequential scan, rolling windows, split methods, and two experimental AR(1) bootstraps. None resolves the serial-correlation concern. |
| [`run.py`](code/run.py) | Command line entry point for a new benchmark run, an example synthetic panel, and all six figures. |

The former weekly scripts were consolidated into these four files. The old source remains available in Git history for auditing the frozen results. The refactor was checked against the old implementation on identical random panels: the test statistic, bootstrap p-value, DGP output, and gated-split decision matched exactly.

## Reproduce

```bash
python -m pip install -r requirements.txt
python code/run.py figures
python code/run.py synthetic-data --dataset jkv --setup break --tau 240 --seed 100000
python code/run.py benchmark --dataset jkv --reps 10 --bootstrap 49 --jobs 1
python code/run.py benchmark --dataset jkv --reps 10 --bootstrap 49 --rho 0.3
```

The small benchmark is a smoke run; it does **not** reproduce the frozen 500-panel result table. A larger full-sample run can use `--reps 500 --bootstrap 200`, but is computationally costly. The benchmark writes resumable per-replication files in ignored `results/cells/`; rerunning the same command uses those cache cells. `python code/run.py figures` uses the committed frozen tables and public input data, so it does not require a new Monte Carlo run.

## Interpretation and data boundary

Under independent simulated monthly errors, the final-seed full-sample 5% test has **3.8% size** on no-break panels and **100% power** against the calibrated full-strength fourth factor at three planted dates. Su and Wang's published power designs keep the factor count fixed and vary loadings; our three-to-four-factor power values are therefore not numerically comparable with theirs.

The JKV-based residuals have median lag-one autocorrelation about 0.63. In a no-break AR(1) stress test with coefficient 0.3, the original bootstrap rejects every tested panel at 5%. This demonstrates a possible false-rejection mechanism; it does **not** establish that every empirical rejection is false. The raw CRSP/Fama–Bliss yield panel is not redistributed. The committed JKV-calibration parameters and synthetic example are sufficient to run the public simulations. Source details and the H.15 sample boundary are in [data/README.md](data/README.md).
