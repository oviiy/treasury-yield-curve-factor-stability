"""Fake yield curves with one change in the number of factors (advisor, 2026-10-06).

WEEK 7 CHANGE: `--factor-model nss` (default) builds the regimes from Nelson-Siegel (3 factors) and Nelson-Siegel-
Svensson (4 factors) loadings fitted to the real panel, as the advisor asked; `--factor-model pca` reproduces the wk6
design (loadings from a PCA of the real panel, which is what wk6 actually used). `--bw-mode/--bw-value` pick the
bandwidth rule of the scan (see run_multibreak.py). Everything else is unchanged.

Original wk6 text: loadings come from a PCA of the real panel (3 factors for the first regime, 4 for the second), factors are AR(1)
with coefficients and innovation variances estimated from the real PCA factors, and the curve is loadings x factors.
The advisor asked for NO extra residual (`--resid none`), but then each regime is exactly rank 3 / rank 4, the
fitted residuals are rounding error (MSE ~1e-31) and the test statistic is numerical noise, so the scan rejects
even inside a pure three-factor regime. `--resid real` (default) adds i.i.d. N(0, s_i^2) noise per maturity, with
s_i the real-data residual s.d. of that regime's PCA fit. The same segmented scan used on the real data is then run
at the chosen ranks.

Setups: "break" (3 factors, then 4 at `--break-frac`), "null3" (3 throughout), "null4" (4 throughout).
Usage: python3 wk7/scripts/sim_regime_break.py --dataset jkv --setup break --reps 100 --ranks 3 4
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

from wk4_common import WK4, JKV_PATH, silverman_th, TSY_LONG_COLS, load_jkv, load_treasury_cmt_long, run_test  # first: pins BLAS to one thread before numpy loads
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from run_multibreak import WindowTester
from segmented_scan import segmented_scan


def fit_pca(y, d):
    """Loadings (N, d) and factors (T, d) of the demeaned panel; factors carry the scale (loadings are unit norm)."""
    x = y - y.mean(0)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return vt[:d].T, u[:, :d] * s[:d]


def maturities(dataset):
    """Maturities in months, in the column order of the loader."""
    if dataset == "jkv":
        return np.array([int(c) for c in pd.read_csv(JKV_PATH, nrows=1).columns[1:]], float)
    return np.array([int(c[1:]) for c in TSY_LONG_COLS], float)


def ns_loadings(tau, lams):
    """Nelson-Siegel (one lambda: level, slope, curvature) or Nelson-Siegel-Svensson (two lambdas: + second curvature)
    loadings, (N, 3) or (N, 4). lambda is per month (Diebold-Li: 0.0609 for maturities in months)."""
    cols = [np.ones_like(tau)]
    l1 = lams[0] * tau
    slope = (1 - np.exp(-l1)) / l1
    cols += [slope, slope - np.exp(-l1)]
    if len(lams) == 2:
        l2 = lams[1] * tau
        cols.append((1 - np.exp(-l2)) / l2 - np.exp(-l2))
    return np.column_stack(cols)


def fit_nss(y, d, tau, lam1=None):
    """Loadings (N, d) and factors (T, d) of the demeaned panel: cross-sectional least squares at every date, with
    lambda(s) chosen to minimize the total squared error (d = 3: Nelson-Siegel; d = 4: Svensson, two lambdas)."""
    x = y - y.mean(0)

    def sse(p):
        lam = np.exp(p)  # lambdas stay positive
        if lam1 is not None:  # d = 4 with lambda1 fixed: p holds only lambda2, so the first three loadings equal the NS ones
            lam = np.array([lam1, lam[0]])
        # keep lambdas in a range where each curvature loading is a distinct hump (lambda -> 0 collapses it to a line)
        if lam.min() < 0.01 or lam.max() > 1 or (d == 4 and abs(lam[0] - lam[1]) < 0.005):
            return 1e12
        load = ns_loadings(tau, lam)
        beta = np.linalg.lstsq(load, x.T, rcond=None)[0]
        return float(((x.T - load @ beta) ** 2).sum())
    starts = [np.log([a]) for a in (0.01, 0.03, 0.12, 0.3)] if lam1 is not None else [np.log([0.06])] if d == 3 else [np.log([a, b]) for a in (0.03, 0.06, 0.12) for b in (0.01, 0.2, 0.4) if a != b]
    best = min((minimize(sse, p0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 600}) for p0 in starts),
               key=lambda r: r.fun)
    lams = np.array([lam1, *np.exp(best.x)]) if lam1 is not None else np.exp(best.x)
    load = ns_loadings(tau, lams)
    f = np.linalg.lstsq(load, x.T, rcond=None)[0].T
    return load, f - f.mean(0), lams


def ar1_params(f):
    """Per-factor AR(1) coefficient and innovation s.d. (OLS with intercept)."""
    rho, sd = np.zeros(f.shape[1]), np.zeros(f.shape[1])
    for j in range(f.shape[1]):
        a, b = f[:-1, j], f[1:, j]
        rho[j] = np.cov(a, b)[0, 1] / np.var(a, ddof=1)
        sd[j] = np.std(b - b.mean() - rho[j] * (a - a.mean()), ddof=2)
    return rho, sd


def simulate_factors(rho, sd, T, rng, burn=200):
    u = rng.standard_normal((T + burn, len(rho))) * sd
    f = np.zeros_like(u)
    for t in range(1, T + burn):
        f[t] = rho * f[t - 1] + u[t]
    return f[burn:]


def make_model(y, factor_model="nss", tau=None, shared_lambda=True):
    """Real-data parameters for both regimes (full-sample fit, fitted once)."""
    model, lam1 = {}, None
    for d in (3, 4):
        if factor_model == "nss":
            lam, f, lambdas = fit_nss(y, d, tau, lam1=lam1 if (d == 4 and shared_lambda) else None)
            lam1 = lambdas[0] if d == 3 else lam1
            print(f"d={d} NS{'S' if d == 4 else ''} lambda(s) per month: {np.round(lambdas, 4)}", flush=True)
        else:
            lam, f = fit_pca(y, d)
        resid_sd = (y - y.mean(0) - f @ lam.T).std(0, ddof=1)
        model[d] = (lam, *ar1_params(f), resid_sd)
    return model


def simulate_curve(model, T, setup, break_month, rng, resid="real"):
    """Returns (T, N) fake yields (demeaned scale; the test standardizes within each window) and the true break."""
    def regime(d, n):
        lam, rho, sd, resid_sd = model[d]
        y = simulate_factors(rho, sd, n, rng) @ lam.T
        if resid == "same4":  # both regimes get the four-factor residual s.d.: no change in noise variance at the break
            resid_sd = model[4][3]
        return y + rng.standard_normal(y.shape) * resid_sd if resid in ("real", "same4") else y
    if setup == "null3":
        return regime(3, T), None
    if setup == "null4":
        return regime(4, T), None
    return np.vstack([regime(3, break_month), regime(4, T - break_month)]), break_month


def one_rep(args):
    y_model, T, setup, break_month, ranks, min_window, restart, alpha, quick, resid, run_k, bw, seed = args
    rng = np.random.default_rng(seed)
    y, true_break = simulate_curve(y_model, T, setup, break_month, rng, resid)
    reps = {"screen_reps": 199, "confirm_reps": 0} if quick else {}
    out = {"seed": seed, "true_break": true_break}
    for d in ranks:
        segs = segmented_scan(T, WindowTester(y, d, alpha=alpha, bw_mode=bw[0], bw_value=bw[1], early_stop=True, **reps), min_window, restart_min_window=restart,
                              run_k=run_k)
        out[f"d{d}"] = [g["break_month"] for g in segs if g["break_month"] is not None]
        # week 7: the full-sample test on the same fake panel (999 draws, Silverman h from the full T), to compare with the scan
        full, _, _ = run_test(y, d, 999, seed + 99, h=None, stop_alpha=alpha)
        out[f"full_p{d}"] = full["bootstrap_p_add_one"]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["jkv", "treasury_cmt"], default="jkv")
    p.add_argument("--setup", choices=["break", "null3", "null4"], default="break")
    p.add_argument("--T", type=int, default=480)
    p.add_argument("--break-frac", type=float, default=0.5)
    p.add_argument("--ranks", type=int, nargs="+", default=[3, 4])
    p.add_argument("--min-window", type=int, default=60)
    p.add_argument("--restart-min-window", type=int, default=60)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--reps", type=int, default=100)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--run-k", type=int, default=1, help="consecutive rejections needed for a break (1 = original rule)")
    p.add_argument("--resid", choices=["real", "none", "same4"], default="real",
                   help="none = advisor's original spec (degenerate test, see module docstring); same4 = the "
                        "four-factor residual s.d. in both regimes (isolates the factor change from a noise change)")
    p.add_argument("--factor-model", choices=["nss", "pca"], default="nss",
                   help="nss = Nelson-Siegel (3) / Svensson (4) loadings (week 7, advisor); pca = the wk6 design")
    p.add_argument("--nss-lambda", choices=["shared", "free"], default="shared",
                   help="shared: the Svensson regime reuses the Nelson-Siegel lambda1, so three factors are nested in four "
                        "(week 7 default); free: each regime fits its own lambda1 (first week-7 run; NOT nested)")
    p.add_argument("--bw-mode", choices=["silverman", "fixed_h", "fixed_th", "scale", "full"], default="silverman",
                   help="full (week 8) = the full-sample T*h of the simulated T in every window")
    p.add_argument("--bw-value", type=float, default=None)
    a = p.parse_args()
    _, y = {"jkv": load_jkv, "treasury_cmt": load_treasury_cmt_long}[a.dataset]()
    model = make_model(y, a.factor_model, maturities(a.dataset), a.nss_lambda == "shared")
    if a.bw_mode == "full" and a.bw_value is None:
        a.bw_value = round(float(silverman_th(a.T, y.shape[1])), 1)  # T = 480: 71.4 (JKV, N = 17), 76.9 (Treasury, N = 8)
    bm = int(round(a.T * a.break_frac))
    tag = f"{a.dataset}_{a.setup}_f{a.break_frac:g}_mw{a.min_window}_restart{a.restart_min_window}"
    tag += {"none": "_noresid", "same4": "_same4", "real": ""}[a.resid] + (f"_k{a.run_k}" if a.run_k != 1 else "")
    tag += "_pca" if a.factor_model == "pca" else ("_nsfree" if a.nss_lambda == "free" else "")
    tag += f"_{a.bw_mode}{a.bw_value:g}" if a.bw_mode != "silverman" else ""
    tag += f"_a{a.alpha:g}" if a.alpha != 0.01 else ""  # week 8: alpha in the name, so runs at 5% / 10% never resume a 1% file
    tag += "_quick" if a.quick else ""
    (WK4 / "results").mkdir(exist_ok=True)
    # each finished replication is appended here, so an interrupted run resumes where it stopped
    partial = WK4 / f"results/simreg_{tag}.partial.jsonl"
    done = [json.loads(l) for l in partial.read_text().splitlines()] if partial.exists() else []
    seen = {r["seed"] for r in done}
    jobs = [(model, a.T, a.setup, bm, a.ranks, a.min_window, a.restart_min_window, a.alpha, a.quick, a.resid, a.run_k, (a.bw_mode, a.bw_value), 7_000 + i)
            for i in range(a.reps) if 7_000 + i not in seen]
    # one process per replication; the window tests inside each scan run serially
    with ProcessPoolExecutor(a.jobs) as pool, partial.open("a") as fh:
        for r in pool.map(one_rep, jobs):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            done.append(r)
    res = sorted(done, key=lambda r: r["seed"])[:a.reps]
    spec = {k: v for k, v in vars(a).items()}
    (WK4 / f"results/simreg_{tag}.json").write_text(json.dumps({"spec": spec, "true_break": bm, "reps": res}, indent=1))
    for d in a.ranks:
        n = [len(r[f"d{d}"]) for r in res]
        fp = np.array([r[f"full_p{d}"] for r in res]) < a.alpha
        scan_any = np.array([len(r[f"d{d}"]) > 0 for r in res])
        print(f"d={d}: full-sample rejects {fp.mean():.2f}; scan finds a break {scan_any.mean():.2f}; "
              f"scan breaks while full sample does not {np.mean(scan_any & ~fp):.2f}; full rejects but scan finds none {np.mean(fp & ~scan_any):.2f}")
        line = f"d={d}: mean #breaks {np.mean(n):.2f}; P(any break) {np.mean([x > 0 for x in n]):.2f}"
        if a.setup == "break":
            # a forward-growing scan dates a break only once enough post-break months make the test reject,
            # so report the lag of the first break at or after (truth - 12) rather than a symmetric window
            pre = np.mean([any(b < bm - 12 for b in r[f"d{d}"]) for r in res])
            lags = [min((b - bm for b in r[f"d{d}"] if b >= bm - 12), default=None) for r in res]
            hit = [x for x in lags if x is not None and x <= 36]
            line += (f"; P(false break before truth-12) {pre:.2f}; P(first break after truth within 36m) "
                     f"{len(hit) / len(res):.2f}; median lag {np.median(hit) if hit else float('nan'):.0f}m")
        print(line)


if __name__ == "__main__":
    main()
