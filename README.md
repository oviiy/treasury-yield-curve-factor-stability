# Treasury yield-curve factor stability

**Working research repository.** We study whether fixed factor loadings adequately describe U.S. Treasury yield curves. The paper applies the full-sample test of Su and Wang (2017) to two yield panels and uses Monte Carlo experiments to check its size and power at Treasury-like dimensions. Dating a change is a secondary extension.

The current research issue is serial correlation in the residuals. Under serially independent simulation errors, the test is reasonably sized and detects a strong change. When simulated errors persist over time, its original bootstrap can reject even when no change was planted. Empirical rejections and break dates therefore remain provisional.

## Read first

- [Working paper (PDF)](paper/working_draft.pdf) and [LaTeX source](paper/working_draft.tex)
- [Key findings](results/key_findings.md) and [detailed simulation tables](results/final_tables.md)
- [Simulation design](figures/simulation_design.png), [serial-correlation stress test](figures/serial_correlation_stress.png), and [scan dating error](figures/scan_lag.png)

## Code and reproduction

`code/su_wang_monthly.py` implements the global and local PCA test and covariance bootstrap. `code/benchmark.py` contains the no-break and 3-to-4-factor data generators, experiment runner, and size/power/dating metrics. `code/methods.py` contains the AR(1) bootstrap trials and alternative dating methods. The remaining modules support those methods and the larger experiments.

Install Python dependencies with `python -m pip install -r requirements.txt`. A small synthetic-only run needs no original yield data:

```bash
python code/run_public_benchmark.py --reps 10 --bootstrap 49
```

For a paper-scale full-sample run, use `--reps 500 --bootstrap 200`; this takes substantially longer. `--rho 0.3` changes the simulated idiosyncratic errors from serially independent to AR(1). The published summary tables are the frozen results of the larger experiments; their per-replication caches are omitted from this compact repository.

## Data boundary

The JKV/Fama–Bliss panel is based on CRSP Treasury data, which are **not included**. The `calibration/` files contain fitted Nelson–Siegel/Svensson loadings, factor persistence and scale parameters, and per-maturity residual standard deviations, so the synthetic experiments can run without raw yields. To reproduce the empirical tests or recalibrate parameters, obtain the source data under the appropriate license and place them as described in the loader paths in `code/wk4_common.py`. The U.S. Treasury CMT series comes from Federal Reserve H.15/Treasury sources; raw copies are also omitted to keep this public repository focused.

The [working paper](paper/working_draft.pdf) is a research draft. The unresolved serial-correlation issue is stated explicitly; numerical rejections should not yet be treated as conclusive evidence of time-varying loadings.
