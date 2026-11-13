"""Su–Wang full-sample J test and covariance bootstrap.

Equation-based implementation, not the authors' MATLAB program. The bootstrap
draws are independent across months; this assumption is a central limitation.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def _paper_kernel_matrices(t, n, h=None):
    if h is None:
        h = (2.35 / np.sqrt(12)) * t**(-.2) * n**(-.1)
    dates = np.arange(1, t+1)
    u = (dates[:, None]-dates[None, :]) / (t*h)  # row s, column evaluation t
    k = .75*np.maximum(1-u*u, 0) / h
    # Page 87: continuous boundary mass; interior is unchanged.
    primitive = lambda z: .75*(z-z**3/3)
    mass = np.ones(t)
    lo = dates < np.floor(t*h)
    hi = dates > t-np.floor(t*h)
    mass[lo] = primitive(1)-primitive(-dates[lo]/(t*h))
    mass[hi] = primitive((1-dates[hi]/t)/h)-primitive(-1)
    k /= mass[None, :]
    v = np.abs(u)
    convolution = np.where(v <= 2, .6-.75*v*v+.375*v**3-(3/160)*v**5, 0)
    return h, k, convolution

def sign_loadings(v):
    return v * np.where(v[0] < 0, -1., 1.)

def fit(x, rank, k):
    t, n = x.shape
    _, singular, vt = np.linalg.svd(x, full_matrices=False)
    v = sign_loadings(vt[:rank].T)
    fixed_loadings = v * (singular[:rank]/np.sqrt(t))
    fixed_factors = (x @ v) / (singular[:rank]/np.sqrt(t))
    fixed_common = fixed_factors @ fixed_loadings.T
    # Batch of N by N covariance matrices is equivalent to weighted T by T PCA.
    # cov[t] = sum_s k[s,t] x_s x_s' / T, via one (T x T) @ (T x N^2) product; avoids a T*T*N*N temporary.
    cov = (k.T @ np.einsum("si,sj->sij", x, x).reshape(t, n * n)).reshape(t, n, n) / t
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    vals = eigenvalues[:, -rank:][:, ::-1]
    vectors = eigenvectors[:, :, -rank:][:, :, ::-1]
    vectors *= np.where(vectors[:, 0:1, :] < 0, -1., 1.)
    if np.min(vals) <= 1e-14:
        raise ValueError("Degenerate local eigenvalue; J cannot be evaluated reliably")
    loading = vectors * np.sqrt(vals)[:, None, :]
    factors = np.einsum("ti,tir->tr", x, vectors) / np.sqrt(vals)
    common = np.einsum("tir,tr->ti", loading, factors)
    return {"fixed_common": fixed_common, "fixed_factors": fixed_factors,
            "fixed_loadings": fixed_loadings, "common": common, "factors": factors,
            "loadings": loading, "eigenvalues": vals}

def statistic(x, fitted, h, k, convolution):
    t, n = x.shape
    f = fitted["factors"]
    f0 = fitted["fixed_factors"]
    e = x-fitted["common"]
    m = float(np.mean((fitted["common"]-fitted["fixed_common"])**2))
    inner = k*(f @ f.T)-(f0 @ f0.T)
    bias = np.sqrt(h)/(t*t*np.sqrt(n))*np.sum(inner**2*np.sum(e*e, axis=1)[:, None])
    sigma_f = f.T @ f / t
    term = convolution**2 * (f @ sigma_f @ f.T)**2 * (e @ e.T)**2
    np.fill_diagonal(term, 0)
    variance = 2*np.sum(term)/(t*t*n*h)
    if variance <= 0 or not np.isfinite(variance):
        raise ValueError("Non-positive studentizing variance")
    j = (t*np.sqrt(n*h)*m-bias)/np.sqrt(variance)
    return {"M": float(m), "bias": float(bias), "variance": float(variance), "J": float(j)}

def bootstrap(x, rank, h, k, convolution, reps, seed, verbose=True):
    t, n = x.shape
    fitted = fit(x, rank, k)
    obs = statistic(x, fitted, h, k, convolution)
    e0 = x-fitted["fixed_common"]
    covariance = (e0.T @ e0/t) * .99**np.abs(np.arange(n)[:, None]-np.arange(n)[None, :])
    eig, vec = np.linalg.eigh(covariance)
    assert eig.min() >= -1e-12
    root = vec @ np.diag(np.sqrt(np.maximum(eig, 0)))
    rng = np.random.default_rng(seed)
    draws = []
    for b in range(reps):
        xb = fitted["fixed_common"] + rng.standard_normal((t,n)) @ root.T
        fb = fit(xb, rank, k)
        draws.append(statistic(xb, fb, h, k, convolution)["J"])
        if verbose and (b+1) % 100 == 0:
            print(f"R={rank}: {b+1}/{reps} bootstrap draws", flush=True)
    draws = np.asarray(draws)
    # Section 4.6 uses the strict upper-tail proportion; also retain add-one for transparency.
    exceedances = int(np.sum(draws >= obs["J"]))
    p_paper = float(np.mean(draws > obs["J"]))
    obs.update({"R": rank, "B": reps, "seed": seed, "exceedances": exceedances,
        "bootstrap_p_paper": p_paper, "bootstrap_p_add_one": (exceedances+1)/(reps+1),
        "critical_95": float(np.quantile(draws,.95)), "critical_90": float(np.quantile(draws,.90)),
        "critical_99": float(np.quantile(draws,.99)),
        "reject_1pct": bool((exceedances+1)/(reps+1) < .01),
        "static_residual_mse": float(np.mean(e0**2)),
        "local_residual_mse": float(np.mean((x-fitted['common'])**2)),
        "static_variance_share": float(1-np.sum(e0**2)/np.sum(x**2)),
        "local_residual_lag1_median": float(np.median([np.corrcoef((x-fitted['common'])[:-1,i],(x-fitted['common'])[1:,i])[0,1] for i in range(n)]))})
    return obs, fitted, draws

def fixed_approximation(common):
    # Descriptive sample counterpart to the infimum in A.9, not a new hypothesis test.
    _, s, _ = np.linalg.svd(common, full_matrices=False)
    return [{"fixed_R": q, "mean_squared_gap": float(np.sum(s[q:]**2)/common.size),
             "unexplained_common_percent": float(100*np.sum(s[q:]**2)/np.sum(s**2))} for q in (3,4,5)]

def silverman_th(t, n):
    """Effective bandwidth T*h in months of the rule of thumb h = (2.35/sqrt(12)) T^-1/5 N^-1/10 (week 8: the
    full-sample bandwidth that every scan window reuses)."""
    return t * (2.35 / np.sqrt(12)) * t ** -.2 * n ** -.1

def kernel_matrices(t, n, h=None):
    """Week 8: _paper_kernel_matrices with the boundary kernel extended to bandwidths wider than half the window.

    The paper's boundary kernel (p. 87, Hong-Li 2005) renormalises K by its mass inside the sample, with a left zone
    r < floor(Th) and a right zone r > T - floor(Th) that never overlap in the paper (T*h < T/2). With the full-sample
    bandwidth in a short window (week 8) the zones overlap and the paper / MATLAB / sw code let the right zone overwrite
    the left, so the weights no longer integrate to one. Best practice (cut-and-normalise; Li-Racine 2007 p. 31, Jones
    1993): divide by the mass on the in-sample part of the support, truncated on both sides,
    int_{-r/(Th)}^{(1-r/T)/h} K(u) du. Dates outside the overlap keep the paper's kernel bit for bit."""
    h, k, convolution = _paper_kernel_matrices(t, n, h)
    dates = np.arange(1, t + 1)
    both = (dates < np.floor(t * h)) & (dates > t - np.floor(t * h))
    if both.any():
        primitive = lambda z: .75 * (z - z ** 3 / 3)
        u = (dates[:, None] - dates[None, both]) / (t * h)
        raw = .75 * np.maximum(1 - u * u, 0) / h
        r = dates[both]
        k[:, both] = raw / (primitive(np.minimum(1, (1 - r / t) / h)) - primitive(np.maximum(-1, -r / (t * h))))[None, :]
    return h, k, convolution

def bootstrap_early_stop(x, rank, h, k, conv, reps, seed, alpha):
    """Week 7 speed-up: bootstrap with the same draws in the same order, stopped as soon as the add-one decision is
    settled. Once (exceedances + 1) / (reps + 1) >= alpha the window cannot reject whatever the remaining draws are, so
    the reject / not-reject decision is exactly that of the full run; only the p-value and critical values are not
    computed (p is reported as the add-one lower bound, critical_99 as NaN)."""
    t, n = x.shape
    fitted = fit(x, rank, k)
    obs = statistic(x, fitted, h, k, conv)
    e0 = x - fitted["fixed_common"]
    covariance = (e0.T @ e0 / t) * .99 ** np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    eig, vec = np.linalg.eigh(covariance)
    root = vec @ np.diag(np.sqrt(np.maximum(eig, 0)))
    rng = np.random.default_rng(seed)
    exceed, used = 0, 0
    for _ in range(reps):
        xb = fitted["fixed_common"] + rng.standard_normal((t, n)) @ root.T
        used += 1
        exceed += statistic(xb, fit(xb, rank, k), h, k, conv)["J"] >= obs["J"]
        if (exceed + 1) / (reps + 1) >= alpha:
            break
    obs.update({"R": rank, "B": reps, "draws_used": used, "exceedances": exceed, "stopped_early": used < reps,
                "bootstrap_p_add_one": (exceed + 1) / (reps + 1), "critical_99": float("nan")})
    return obs, fitted, None

def run_test(y, rank, reps, seed, h=None, scale_from=None, stop_alpha=None):
    """Standardize y (own moments, or scale_from's as a control) and run the Su-Wang bootstrap test.
    stop_alpha (week 7): stop the bootstrap once the decision at that level is settled (bootstrap_early_stop)."""
    ref = y if scale_from is None else scale_from
    x = (y - ref.mean(0)) / ref.std(0, ddof=1)
    t, n = x.shape
    h_used, k, conv = kernel_matrices(t, n, h)  # week 8: renormalised boundary kernel (same as sw unless T*h > t/2)
    if stop_alpha is not None:
        obs, fitted, draws = bootstrap_early_stop(x, rank, h_used, k, conv, reps, seed, stop_alpha)
    else:
        obs, fitted, draws = bootstrap(x, rank, h_used, k, conv, reps, seed, verbose=False)
    obs.update({"h": float(h_used), "Th_months": float(t * h_used), "T": t, "N": n})
    return obs, fitted, draws
