"""Shared loaders and a one-call Su-Wang test wrapper for week-4 scripts."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)  # spurious macOS Accelerate matmul warnings

ROOT = Path(__file__).resolve().parents[1]  # handover folder: code/, data/, results/
# NB: kept the name WK4 (rather than renaming every importer) but pointed at wk7 --
# the wk4-copied scripts all wrote into wk4/results and wk4/figures otherwise,
# which would have silently overwritten earlier weeks' delivered results.
WK4 = ROOT
sys.path.insert(0, str(ROOT / "code"))
import su_wang_monthly as sw  # noqa: E402

JKV_PATH = ROOT / "data/UnsmFB_70-09.csv"
TSY_PATH = ROOT / "data/treasury_par_yields_monthly_1990_2026.csv"  # not included; unused here
TSY_CMT_H15_PATH = ROOT / "data/treasury_cmt_h15_monthly_1962_2026.csv"
# Only maturities with no gaps from 1990-01 onward (m1 starts 2001, m240 1993-10, m360 has a 2002-06 gap).
TSY_LONG_COLS = ["m3", "m6", "m12", "m24", "m36", "m60", "m84", "m120"]
TREASURY_METHOD_CHANGE = "2021-12-06"


def load_jkv(start=None, end=None):
    df = pd.read_csv(JKV_PATH)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    if start is not None:
        df = df[df["date"] >= start]
    if end is not None:
        df = df[df["date"] <= end]
    df = df.reset_index(drop=True)
    return df["date"], df.drop(columns="date").to_numpy(float)


def load_treasury(cols=TSY_LONG_COLS, start="1990-01-01", end="2026-08-31"):
    df = pd.read_csv(TSY_PATH, parse_dates=["date"])
    df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
    missing = df[cols].isna().sum()
    if missing.any():
        raise ValueError(f"missing values: {missing[missing > 0].to_dict()}")
    return df["date"], df[cols].to_numpy(float)


def load_treasury_cmt_long(cols=TSY_LONG_COLS, start="1981-09-01", end="2026-08-31"):
    """Official Treasury CMT month-end yields from Fed H.15 (identical to treasury.gov par yields since 1990)."""
    df = pd.read_csv(TSY_CMT_H15_PATH, parse_dates=["date"])
    df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
    missing = df[cols].isna().sum()
    if missing.any():
        raise ValueError(f"missing values: {missing[missing > 0].to_dict()}")
    return df["date"], df[cols].to_numpy(float)


def silverman_th(t, n):
    """Effective bandwidth T*h in months of the rule of thumb h = (2.35/sqrt(12)) T^-1/5 N^-1/10 (week 8: the
    full-sample bandwidth that every scan window reuses)."""
    return t * (2.35 / np.sqrt(12)) * t ** -.2 * n ** -.1


def kernel_matrices(t, n, h=None):
    """Week 8: sw.kernel_matrices with the boundary kernel extended to bandwidths wider than half the window.

    The paper's boundary kernel (p. 87, Hong-Li 2005) renormalises K by its mass inside the sample, with a left zone
    r < floor(Th) and a right zone r > T - floor(Th) that never overlap in the paper (T*h < T/2). With the full-sample
    bandwidth in a short window (week 8) the zones overlap and the paper / MATLAB / sw code let the right zone overwrite
    the left, so the weights no longer integrate to one. Best practice (cut-and-normalise; Li-Racine 2007 p. 31, Jones
    1993): divide by the mass on the in-sample part of the support, truncated on both sides,
    int_{-r/(Th)}^{(1-r/T)/h} K(u) du. Dates outside the overlap keep the paper's kernel bit for bit."""
    h, k, convolution = sw.kernel_matrices(t, n, h)
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
    """Week 7 speed-up: sw.bootstrap with the same draws in the same order, stopped as soon as the add-one decision is
    settled. Once (exceedances + 1) / (reps + 1) >= alpha the window cannot reject whatever the remaining draws are, so
    the reject / not-reject decision is exactly that of the full run; only the p-value and critical values are not
    computed (p is reported as the add-one lower bound, critical_99 as NaN)."""
    t, n = x.shape
    fitted = sw.fit(x, rank, k)
    obs = sw.statistic(x, fitted, h, k, conv)
    e0 = x - fitted["fixed_common"]
    covariance = (e0.T @ e0 / t) * .99 ** np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    eig, vec = np.linalg.eigh(covariance)
    root = vec @ np.diag(np.sqrt(np.maximum(eig, 0)))
    rng = np.random.default_rng(seed)
    exceed, used = 0, 0
    for _ in range(reps):
        xb = fitted["fixed_common"] + rng.standard_normal((t, n)) @ root.T
        used += 1
        exceed += sw.statistic(xb, sw.fit(xb, rank, k), h, k, conv)["J"] >= obs["J"]
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
        obs, fitted, draws = sw.bootstrap(x, rank, h_used, k, conv, reps, seed, verbose=False)
    obs.update({"h": float(h_used), "Th_months": float(t * h_used), "T": t, "N": n})
    return obs, fitted, draws
