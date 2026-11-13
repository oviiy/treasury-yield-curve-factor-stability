"""Su and Wang (2017), Sections 3.1, 4.2--4.6, on monthly yield levels.

Feasible J statistic and covariance-shrinkage Gaussian bootstrap. This is a
new equation-based implementation, not the authors' original program.
The second moment Sigma_F_hat is F_hat' F_hat / T. Signs follow the first
maturity row convention on p. 87. No Procrustes rotation enters the test.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def kernel_matrices(t, n, h=None):
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


def main(reps=9999):
    frame = pd.read_csv(ROOT / "data/derived/monthly_yields_2006_2026.csv")
    y = frame.iloc[:,1:].to_numpy()
    x = (y-y.mean(0))/y.std(0,ddof=1)
    t,n = x.shape
    h,k,convolution = kernel_matrices(t,n)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    rows = []
    fit3 = None
    for rank in (3,4,5):
        row, fitted, draws = bootstrap(x,rank,h,k,convolution,reps,20170000+rank)
        rows.append(row)
        np.save(out / f"bootstrap_J_R{rank}.npy",draws)
        if rank==3:
            fit3=fitted
    np.savez(out / "three_factor_estimates.npz", **fit3)
    summary = {"sample": {"T":t,"N":n,"start":frame.date.iloc[0],"end":frame.date.iloc[-1],
        "h":h,"Th_months":t*h,"maturity_columns":frame.columns[1:].tolist()},
        "tests":rows,"decisions": {"alpha":.01, "p_question_1":rows[0]["bootstrap_p_add_one"], "p_question_2":max(r["bootstrap_p_add_one"] for r in rows[1:]), "p_overall":max(r["bootstrap_p_add_one"] for r in rows), "question_2_rejected":all(r["reject_1pct"] for r in rows[1:]), "overall_chain_passed":all(r["reject_1pct"] for r in rows)},"fixed_approximation_of_local_three_factor_common":fixed_approximation(fit3["common"]),
        "method": "Su-Wang feasible J, p.92; Section 4.6 Gaussian covariance bootstrap, epsilon=0.01; first-maturity sign convention; Sigma_F_hat=F_hat.T@F_hat/T",
        "qualification": "New equation-based implementation. Large-N and martingale-error assumptions are not certified by this monthly N=17 application. A.9 is an asymptotic assumption, not proven by these sample results."}
    (out / "monthly_results.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / "rank_tests.csv",index=False)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--reps",type=int,default=9999)
    args=parser.parse_args()
    if args.reps < 1: parser.error("--reps must be positive")
    main(args.reps)
