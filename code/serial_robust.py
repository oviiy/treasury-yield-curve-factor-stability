"""Serial-correlation-robust Su–Wang loading test: sieve-AR GLS-J.

The original Su–Wang (2017) bootstrap (`sw_test.py`) draws errors independently
month by month. Yield-curve residuals are persistent, and under AR errors that
test rejects almost every panel with constant loadings. This module keeps the
Su–Wang null, lambda_i(t) = lambda_i, and the kernel-local versus full-sample
loading comparison, and changes three things:

1. Filter. Fit PCA factors, choose one AR order p for all maturities by pooled
   AIC (p <= pmax), fit an AR(p) to each maturity's residual, and quasi-difference
   yields and factors (Breitung and Tenhofen 2011, GLS for factor models).
2. Compare. On the filtered data, estimate kernel-local loadings lambda_i(t) and
   full-sample loadings lambda_i by (weighted) least squares and form
   M_GLS = mean_{t,i} [(lambda_i(t) - lambda_i)' z_it]^2 / sigma^2_{u,i}
   (the regression form of Su and Wang 2020). Bandwidth h = 2 x the Su–Wang rule
   of thumb, fixed in advance.
3. Calibrate. Wild bootstrap (one Rademacher sign per month, shared by all
   maturities) on the AR innovations, rebuild the errors through the fitted AR(p)
   recursion with p held fixed, re-estimate everything in every draw (sieve
   bootstrap; Kreiss 1992, Bühlmann 1997).

This is exploratory Monte Carlo methodology. There is no asymptotic validity
proof yet. Variants that were tried and dropped (dependent wild bootstrap on the
original J, quasi-differenced original J, Kendall bias correction, iterated
factors, local variance weights, two-half weighted PCA) are documented in
`results/serial_robust_tables.md`, not implemented here.

The DGP below reproduces the calibrated serial-error scenarios exactly
(same random-number order and seeds) as the exploratory runs summarised in
`results/serial/`; equality was checked panel by panel against the original
workstream code.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from simulation import simulate_factors
from sw_test import kernel_matrices, sign_loadings

ROOT = Path(__file__).resolve().parents[1]
ALPHA = .05
T_SIM = 240
SEED_BASE = 7_000_000
# Calibrations used by the serial-error experiments: JKV 17 maturities 1970–2009 (same file as
# week 9) and the eight-maturity Treasury par curve 1990–2009 (identical to H.15 CMT in that window).
CALIBRATION = {"JKV17": "model_jkv.npz", "Treasury8": "model_treasury8_1990_2009.npz"}


# ------------------------------------------------------------------ statistic
def pca_factors(x, rank):
    """Uncentred PCA as in the exploratory code: F'F/T = I, loadings first-row positive."""
    t = x.shape[0]
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    v = sign_loadings(vt[:rank].T)
    f = (x @ v) / (s[:rank] / np.sqrt(t))
    lam = v * (s[:rank] / np.sqrt(t))
    return f, lam


def ar_fit(e, p):
    """Per-maturity AR(p) by OLS on residual columns; returns (N, p) coefficients (AR(1) clipped to +-.95)."""
    t, n = e.shape
    coef = np.zeros((n, p))
    for i in range(n):
        y = e[p:, i]
        X = np.column_stack([e[p - j - 1:t - j - 1, i] for j in range(p)])
        coef[i] = np.linalg.lstsq(X, y, rcond=None)[0]
    if p == 1:
        coef = np.clip(coef, -.95, .95)
    return coef


def quasi_diff(a, coef):
    """a: (T, N) or (T, N, r); coef (N, p). Returns the filtered rows p..T-1."""
    p, t = coef.shape[1], a.shape[0]
    out = a[p:].copy()
    for j in range(p):
        c = coef[:, j] if a.ndim == 2 else coef[:, j][:, None]
        out = out - c * a[p - j - 1:t - j - 1]
    return out


def select_ar_order(e, pmax=6, crit="aic"):
    """Sieve step: one AR order for all maturities minimising pooled AIC (or BIC) on a common effective sample."""
    t, n = e.shape
    best = (np.inf, 1)
    for p in range(1, pmax + 1):
        u = quasi_diff(e, ar_fit(e, p))[pmax - p:]
        pen = np.log(len(u)) if crit == "bic" else 2.0
        ic = np.sum(np.log(np.mean(u ** 2, 0))) + n * p * pen / len(u)
        best = min(best, (ic, p))
    return best[1]


def gls_statistic(x, rank, k, p):
    """M_GLS on AR(p)-filtered data; k is the (T-p) x (T-p) Su–Wang kernel matrix k[s, t]."""
    n = x.shape[1]
    f, lam = pca_factors(x, rank)
    coef = ar_fit(x - f @ lam.T, p)
    xt = quasi_diff(x, coef)                                     # (T-p, N)
    z = quasi_diff(np.repeat(f[:, None, :], n, axis=1), coef)    # (T-p, N, r)
    zz = np.einsum("tir,tiq->tirq", z, z)
    zx = z * xt[:, :, None]
    lam_g = np.linalg.solve(zz.sum(0), zx.sum(0)[..., None])[..., 0]          # full-sample, (N, r)
    lam_l = np.linalg.solve(np.einsum("st,sirq->tirq", k, zz),
                            np.einsum("st,sir->tir", k, zx)[..., None])[..., 0]  # local, (T-p, N, r)
    u = xt - np.einsum("tir,ir->ti", z, lam_g)
    diff = np.einsum("tir,tir->ti", lam_l - lam_g[None], z)
    stat = float(np.mean(diff ** 2 / np.mean(u ** 2, 0)))
    return {"stat": stat, "f": f, "coef": coef, "lam_g": lam_g, "u": u,
            "e0": (x - f @ lam_g.T)[:p], "contrib_t": np.mean(diff ** 2 / np.mean(u ** 2, 0), 1)}


def gls_j_test(x, rank, boot=199, seed=0, p="aic", pmax=6, hmult=2.0, early=True):
    """Sieve-AR GLS-J test of constant loadings on a standardised (T, N) panel.

    p: an integer AR order, or "aic"/"bic" to choose it once on the data (then fixed in the bootstrap).
    early: stop the bootstrap once the 5% decision is settled (exact decision; p is then an add-one lower bound).
    Returns the statistic, add-one bootstrap p-value, decision at 5%, draws used and the AR order."""
    t, n = x.shape
    if p in ("aic", "bic"):
        f0, l0 = pca_factors(x, rank)
        p = select_ar_order(x - f0 @ l0.T, pmax=pmax, crit=p)
    h0 = (2.35 / np.sqrt(12)) * (t - p) ** (-.2) * n ** (-.1)
    _, k, _ = kernel_matrices(t - p, n, h0 * hmult)
    obs = gls_statistic(x, rank, k, p)
    common = obs["f"] @ obs["lam_g"].T
    u = obs["u"] - obs["u"].mean(0)
    coef = obs["coef"]
    rng = np.random.default_rng(seed)
    exceed = used = 0
    for _ in range(boot):
        us = u * rng.choice((-1., 1.), size=t - p)[:, None]
        es = np.empty((t, n))
        es[:p] = obs["e0"]
        for s in range(p, t):
            es[s] = us[s - p] + sum(coef[:, j] * es[s - j - 1] for j in range(p))
        exceed += gls_statistic(common + es, rank, k, p)["stat"] >= obs["stat"]
        used += 1
        if early and (exceed + 1) / (boot + 1) >= ALPHA:
            break
    pval = (exceed + 1) / (boot + 1)
    return {"stat": obs["stat"], "p": pval, "reject": pval < ALPHA, "draws": used, "p_ar": int(p),
            "h": float(h0 * hmult)}


# ------------------------------------------------------------------ BT-GLS J (wk13, 2026-11-26)
# Breitung and Tenhofen (2011) two-step GLS inside the Su–Wang test. gls_j_test above uses the BT
# *loading* step (AR(p) quasi-differencing) with PC factors and an uncentred M_GLS. bt_gls_j_test adds
#   (a) the BT factor step: F_t from the cross-section regression of x_it - sum_j c_ij e_i,t-j on
#       lambda_i with weights 1/omega_i^2 (innovation variance), after GLS loadings; and
#   (b) Su–Wang centring/standardisation: d_it = sum_s C^i_ts u_is with
#       C^i_ts = z_it'(A_it^-1 k_st - A_i^-1) z_is, M = sum_{s,r} D_sr, D_sr = sum_i W^i_sr v_is v_ir,
#       W^i = C^i'C^i, v = u/omega; J = (M - tr D) / sqrt(2 sum_{s!=r} D_sr^2), N(0,1) one-sided.
# Calibration is the same wild sieve bootstrap. The asymptotic N(0,1) p-value is reported but
# over-rejects at N = 8/17 under ARMA/measurement-error nulls; use the bootstrap p-value.
def bt_factors(x, rank, p, iters=1):
    """PC start, then BT GLS rounds: AR(p) per maturity, GLS loadings, weighted GLS factors."""
    t, n = x.shape
    f, lam = pca_factors(x, rank)
    for _ in range(iters):
        coef = ar_fit(x - f @ lam.T, p)
        xt = quasi_diff(x, coef)
        z = quasi_diff(np.repeat(f[:, None, :], n, axis=1), coef)
        lam = np.linalg.solve(np.einsum("tir,tiq->irq", z, z),
                              np.einsum("tir,ti->ir", z, xt)[..., None])[..., 0]
        e = x - f @ lam.T
        w = lam / np.var(quasi_diff(e, coef), axis=0)[:, None]
        ystar = x.copy()
        for j in range(p):
            ystar[p:] -= coef[:, j] * e[p - j - 1:t - j - 1]
        f = ystar @ np.linalg.solve(lam.T @ w, w.T).T
        f = f - f.mean(0)
    return f


def bt_statistic(x, f, coef, k):
    """Centred and standardised GLS form of the Su–Wang statistic for given factors and AR coefficients."""
    n = x.shape[1]
    xt = quasi_diff(x, coef)
    z = quasi_diff(np.repeat(f[:, None, :], n, axis=1), coef)
    zz = np.einsum("tir,tiq->tirq", z, z)
    A = zz.sum(0)
    lam_g = np.linalg.solve(A, np.einsum("tir,ti->ir", z, xt)[..., None])[..., 0]
    u = xt - np.einsum("tir,ir->ti", z, lam_g)
    P = np.linalg.solve(np.einsum("st,sirq->tirq", k, zz), z[..., None])[..., 0]
    Q = np.linalg.solve(A[None], z[..., None])[..., 0]
    C = np.einsum("tir,sir->its", P, z) * k.T[None] - np.einsum("tir,sir->its", Q, z)
    Cv = C * (u / np.sqrt(np.mean(u ** 2, 0))).T[:, None, :]
    m = float(np.sum(Cv.sum(2) ** 2))
    M2 = Cv.reshape(-1, Cv.shape[2])
    D = M2.T @ M2
    bias = float(np.trace(D))
    np.fill_diagonal(D, 0.)
    return {"J": (m - bias) / np.sqrt(2. * float(np.sum(D ** 2))), "M": m / u.size, "u": u}


def _bt_prepare(x, rank, p, factors, iters):
    f = pca_factors(x, rank)[0] if factors == "pca" else bt_factors(x, rank, p, iters)
    lam = np.linalg.lstsq(f, x, rcond=None)[0].T
    return f, lam, ar_fit(x - f @ lam.T, p)


def bt_gls_j_test(x, rank, boot=199, seed=0, p="aic", pmax=6, hmult=2.0, factors="bt", iters=1, early=True):
    """BT-GLS J test. Same AR-order rule, bandwidth and wild sieve bootstrap as gls_j_test.
    factors="bt": BT GLS factors (recommended); "pca": PC factors (isolates the centring)."""
    from scipy.stats import norm
    t, n = x.shape
    if p in ("aic", "bic"):
        f0, l0 = pca_factors(x, rank)
        p = select_ar_order(x - f0 @ l0.T, pmax=pmax, crit=p)
    h0 = (2.35 / np.sqrt(12)) * (t - p) ** (-.2) * n ** (-.1)
    _, k, _ = kernel_matrices(t - p, n, h0 * hmult)
    f, lam, coef = _bt_prepare(x, rank, p, factors, iters)
    obs = bt_statistic(x, f, coef, k)
    common = f @ lam.T
    e0 = (x - common)[:p]
    u = obs["u"] - obs["u"].mean(0)
    rng = np.random.default_rng(seed)
    exceed = used = 0
    for _ in range(boot):
        us = u * rng.choice((-1., 1.), size=t - p)[:, None]
        es = np.empty((t, n))
        es[:p] = e0
        for s in range(p, t):
            es[s] = us[s - p] + sum(coef[:, j] * es[s - j - 1] for j in range(p))
        xb = common + es
        fb, _, cb = _bt_prepare(xb, rank, p, factors, iters)
        exceed += bt_statistic(xb, fb, cb, k)["J"] >= obs["J"]
        used += 1
        if early and (exceed + 1) / (boot + 1) >= ALPHA:
            break
    pval = (exceed + 1) / (boot + 1)
    return {"stat": obs["J"], "p": pval, "p_asy": float(norm.sf(obs["J"])), "reject": pval < ALPHA,
            "draws": used, "p_ar": int(p), "h": float(h0 * hmult)}


# ------------------------------------------------------------------ DGP
# name: (error spec, alternative or None, strength, volatility after T/2, cross-maturity correlation, true rank)
SCENARIOS = {
    "null_iid":        (0.0, None, 0, 1.0, 0.0, 3),
    "null_ar03":       (0.3, None, 0, 1.0, 0.0, 3),
    "null_ar06":       (0.6, None, 0, 1.0, 0.0, 3),
    "null_hetero":     ("het", None, 0, 1.0, 0.0, 3),
    "null_vol15_ar03": (0.3, None, 0, 1.5, 0.0, 3),
    "null_cross_ar06": (0.6, None, 0, 1.0, 0.5, 3),
    "null_r4_ar06":    (0.6, None, 0, 1.0, 0.0, 4),
    "alt_load_ar03":   (0.3, "loading_shift", .5, 1.0, 0.0, 3),
    "alt_load_ar06":   (0.6, "loading_shift", .5, 1.0, 0.0, 3),
    "alt_load_ar06_s1": (0.6, "loading_shift", 1., 1.0, 0.0, 3),
    "alt_fac_ar03":    (0.3, "factor_arrival", .5, 1.0, 0.0, 3),
    "alt_fac_ar06_s1": (0.6, "factor_arrival", 1., 1.0, 0.0, 3),
    "null_ar075":      (0.75, None, 0, 1.0, 0.0, 3),
    "null_r4_iid":     (0.0, None, 0, 1.0, 0.0, 4),
    "null_r4_hetero":  ("het", None, 0, 1.0, 0.0, 4),
    "alt_load_ar03_s1": (0.3, "loading_shift", 1., 1.0, 0.0, 3),
    "alt_load_hetero": ("het", "loading_shift", .5, 1.0, 0.0, 3),
    "alt_load_hetero_s1": ("het", "loading_shift", 1., 1.0, 0.0, 3),
    "alt_load_vol_s1": (0.3, "loading_shift", 1., 1.5, 0.0, 3),
    "alt_fac_ar03_s1": (0.3, "factor_arrival", 1., 1.0, 0.0, 3),
    "null_ar09":       (0.9, None, 0, 1.0, 0.0, 3),
    "null_ar2":        (("ar2", .5, .25), None, 0, 1.0, 0.0, 3),
    "null_arma11":     (("arma", .8, -.4), None, 0, 1.0, 0.0, 3),
    "null_meas":       (("meas", .95, .5), None, 0, 1.0, 0.0, 3),
    "alt_load_ar06_s025": (0.6, "loading_shift", .25, 1.0, 0.0, 3),
    "alt_load_ar06_s075": (0.6, "loading_shift", .75, 1.0, 0.0, 3),
    "alt_load_hetero_s075": ("het", "loading_shift", .75, 1.0, 0.0, 3),
    "alt_load_arma11_s1": (("arma", .8, -.4), "loading_shift", 1., 1.0, 0.0, 3),
    "null_crosschg_ar06": (0.6, None, 0, 1.0, ("switch", .5), 3),
    "null_volslope_ar03": (0.3, None, 0, "slope", 0.0, 3),
    "null_volslope_ar06": (0.6, None, 0, "slope", 0.0, 3),
    "alt_load_crosschg_s1": (0.6, "loading_shift", 1., 1.0, ("switch", .5), 3),
    "alt_load_iid":    (0.0, "loading_shift", .5, 1.0, 0.0, 3),
}  # order matters: the scenario index enters the seed


def load_serial_model(dataset):
    with np.load(ROOT / "calibration" / CALIBRATION[dataset]) as z:
        return {d: (z[f"lam{d}"], z[f"rho{d}"], z[f"sd{d}"], z[f"resid{d}"]) for d in (3, 4)}


def serial_panel(model, scenario, t, seed):
    """Calibrated yields with persistent, possibly misspecified or heteroskedastic errors.

    loading_shift: fixed three factors; from T/2 the third factor also loads on the fourth NSS shape
        (a change in loadings with the factor count unchanged). factor_arrival: a fourth factor switches on at T/2.
    Error specs: AR(1) rho, "het" (rho 0.2..0.7 across maturities), AR(2), ARMA(1,1), persistent AR(0.95) plus
        measurement noise; all scaled to unit marginal SD times the calibrated residual SD.
    vol "slope": after T/2 the noise SD multiplier rises linearly from 1 (shortest) to 2 (longest maturity).
    cross ("switch", c): innovations become Toeplitz-correlated c^|i-j| only after T/2."""
    rho_spec, kind, strength, vol, cross, true_rank = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    load3, rho3, sd3, noise_sd = model[3]
    load4, rho4, sd4, _ = model[4]
    n = load3.shape[0]
    if true_rank == 4:
        f = simulate_factors(rho4, sd4, t, rng)
        common = f @ load4.T
    else:
        f = simulate_factors(rho3, sd3, t, rng)
        common = f @ load3.T
    f4 = simulate_factors(rho4[3:], sd4[3:], t, rng)[:, 0] * max(strength, 1e-12)
    after = (np.arange(t) >= t // 2).astype(float)
    if kind == "factor_arrival":
        common = common + np.outer(f4 * after, load4[:, 3])
    elif kind == "loading_shift":
        scale = np.std(f4, ddof=1) / np.std(f[:, 2], ddof=1)
        common = common + np.outer(f[:, 2] * scale * after, load4[:, 3])
    toeplitz = lambda c: c ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    z = rng.standard_normal((t + 100, n))
    if isinstance(cross, tuple):
        z[100 + t // 2:] = z[100 + t // 2:] @ np.linalg.cholesky(toeplitz(cross[1])).T
    elif cross:
        z = z @ np.linalg.cholesky(toeplitz(cross)).T
    e = np.zeros_like(z)
    if isinstance(rho_spec, tuple):
        kind_e = rho_spec[0]
        if kind_e == "ar2":
            a1, a2 = rho_spec[1:]
            e[0], e[1] = z[0], z[1]
            for s in range(2, len(z)):
                e[s] = a1 * e[s - 1] + a2 * e[s - 2] + z[s]
        elif kind_e == "arma":
            phi, th = rho_spec[1:]
            e[0] = z[0]
            for s in range(1, len(z)):
                e[s] = phi * e[s - 1] + z[s] + th * z[s - 1]
        elif kind_e == "meas":
            phi, share = rho_spec[1:]
            w = rng.standard_normal((t + 100, n))
            a = np.zeros_like(z)
            a[0] = z[0]
            for s in range(1, len(z)):
                a[s] = phi * a[s - 1] + np.sqrt(1 - phi ** 2) * z[s]
            e = np.sqrt(share) * a + np.sqrt(1 - share) * w
        e = e / e[100:].std(0)
    else:
        rho = np.linspace(.2, .7, n) if rho_spec == "het" else np.full(n, float(rho_spec))
        e[0] = z[0]
        for s in range(1, len(z)):
            e[s] = rho * e[s - 1] + np.sqrt(1 - rho ** 2) * z[s]
    e = e[100:] * noise_sd
    if vol == "slope":
        e *= 1.0 + np.outer(after, np.linspace(0.0, 1.0, n))
    else:
        e *= np.where(after > 0, vol, 1.0)[:, None]
    return common + e, true_rank


def scenario_seed(scenario, rep):
    return SEED_BASE + 10_000 * list(SCENARIOS).index(scenario) + rep


def standardise(y):
    return (y - y.mean(0)) / y.std(0, ddof=1)


# ------------------------------------------------------------------ Monte Carlo runner
def _one(args):
    dataset, scenario, rep, boot, kw = args
    seed = scenario_seed(scenario, rep)
    y, rank = serial_panel(load_serial_model(dataset), scenario, T_SIM, seed)
    t0 = time.time()
    kw2 = dict(kw)
    test = bt_gls_j_test if kw2.pop("method", "sieve") == "bt" else gls_j_test
    r = test(standardise(y), rank, boot, seed + 13, **kw2)
    return {"dataset": dataset, "scenario": scenario, "rep": rep, "B": boot, "fit_rank": rank,
            "reject": bool(r["reject"]), "p_lb": r["p"], "stat": r["stat"], "draws": r["draws"],
            "p_ar": r["p_ar"], "sec": round(time.time() - t0, 2), **kw}


def run_serial_cells(datasets, scenarios, reps, boot=199, jobs=4, out=None, **kw):
    """Rejection rate (%) of sieve-AR GLS-J for each dataset x scenario; resumable JSONL in results/cells/."""
    out = out or ROOT / "results/cells/serial_panels.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    key = lambda d: (d["dataset"], d["scenario"], d["rep"], d["B"], json.dumps({k: d.get(k) for k in kw}, sort_keys=True))
    done = {}
    if out.exists():
        for line in out.read_text().splitlines():
            d = json.loads(line)
            done[key(d)] = d
    tasks = [(ds, sc, rep, boot, kw) for rep in range(reps) for sc in scenarios for ds in datasets]
    todo = [a for a in tasks if key({"dataset": a[0], "scenario": a[1], "rep": a[2], "B": a[3], **kw}) not in done]
    if todo:
        with ProcessPoolExecutor(jobs) as pool, out.open("a") as fh:
            for r in pool.map(_one, todo, chunksize=1):
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                done[key(r)] = r
    rows = []
    for ds in datasets:
        for sc in scenarios:
            rej = [done[key({"dataset": ds, "scenario": sc, "rep": i, "B": boot, **kw})]["reject"] for i in range(reps)]
            rate = 100 * float(np.mean(rej))
            rows.append({"dataset": ds, "scenario": sc, "reps": reps, "reject_pct": rate,
                         "mc_se_pts": float(np.sqrt(rate * (100 - rate) / reps))})
    return rows


# ------------------------------------------------------------------ real data
CMT8 = ["m3", "m6", "m12", "m24", "m36", "m60", "m84", "m120"]


def treasury8_samples():
    """Eight-maturity Treasury par curve from the committed H.15 CMT file; complete months, through August 2026."""
    import pandas as pd
    df = pd.read_csv(ROOT / "data/treasury_cmt_h15_monthly_1962_2026.csv", parse_dates=["date"])
    df = df[(df.date >= "1990-01-01") & (df.date <= "2026-08-31")][["date"] + CMT8].dropna()
    return {"Treasury8 1990-2009": df[df.date <= "2009-12-31"],
            "Treasury8 2010-2026": df[df.date >= "2010-01-01"],
            "Treasury8 1990-2026": df}


def empirical_test(y, rank, boot=999, seed=20261122, pmax=6, method="sieve"):
    """Sieve-AR GLS-J (method="sieve") or BT-GLS J (method="bt") on a real (T, N) yield panel in levels;
    no early stopping, B=999 by default. The PCA is uncentred on the raw yields, as in the exploratory run."""
    if method == "bt":
        return bt_gls_j_test(np.asarray(y, float), rank, boot, 20261126 + rank, pmax=pmax, early=False)
    return gls_j_test(np.asarray(y, float), rank, boot, seed + rank, p="aic", pmax=pmax, hmult=2.0, early=False)
