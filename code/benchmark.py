"""Week 9 evaluation harness: DGPs, break-detection methods and size / power / location metrics.

Design rules (advisor, 2026-10-28):
  * size is measured ONLY on a DGP that satisfies the null (3 factors, constant loadings, no break);
  * power and location accuracy are measured ONLY on a DGP with a break (3 -> 4 factors at tau);
  * everything that scores a method (this file's DGPs, metrics and seed blocks) is frozen after the Phase 0 audit;
    exploration changes methods and their parameters, never the scoring.

A method is a callable `m(y) -> {"p": full-sample bootstrap p or None, "breaks": [month, ...]}`.
`legacy_panel` reproduces the wk8 generator bit for bit (regression tests); `panel` is the week-9 DGP.
"""
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

W9 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W9 / "code"))
from wk4_common import load_jkv, load_treasury_cmt_long, run_test, silverman_th  # noqa: E402  (pins BLAS to 1 thread first)
import numpy as np  # noqa: E402
import sim_regime_break as sim  # noqa: E402
from run_multibreak import WindowTester  # noqa: E402
from segmented_scan import segmented_scan  # noqa: E402

# Seed blocks never overlap: tune (explore), validate (pick), final (report). legacy = the wk8 seeds 7000+i.
SEEDS = {"legacy": 7_000, "tune": 100_000, "validate": 200_000, "final": 300_000}
BLOCK = 50_000


def seeds(block, n, start=0):
    assert start + n <= BLOCK, "seed block overflow"
    return [SEEDS[block] + start + i for i in range(n)]


# ---------------------------------------------------------------- data model
def load_model(dataset="jkv"):
    """Load public calibration parameters; raw yields are optional and not redistributed."""
    cache = W9 / f"calibration/model_{dataset}.npz"
    if cache.exists():
        z = np.load(cache)
        return {d: (z[f"lam{d}"], z[f"rho{d}"], z[f"sd{d}"], z[f"resid{d}"]) for d in (3, 4)}, None
    _, y = {"jkv": load_jkv, "treasury_cmt": load_treasury_cmt_long}[dataset]()
    model = sim.make_model(y, "nss", sim.maturities(dataset), True)
    cache.parent.mkdir(exist_ok=True)
    np.savez(cache, **{f"{k}{d}": v for d in (3, 4) for k, v in zip(("lam", "rho", "sd", "resid"), model[d])})
    return model, y


# ---------------------------------------------------------------- DGPs
def legacy_panel(model, T, setup, tau, seed):
    """The wk8 generator (independent regimes, each with its own AR(1) factors and residual s.d.)."""
    return sim.simulate_curve(model, T, setup, tau, np.random.default_rng(seed), "real")[0]


def panel(model, T, setup, tau, seed, strength=1.0, resid_rho=0.0, return_signal=False, resid_from=3):
    """Week-9 DGP: one continuous factor path, so the break is ONLY the arrival of a fourth factor.

    f1..f3: AR(1) with the NS (d=3) fit's loadings, persistence and innovation s.d., for the whole sample;
    f4: AR(1) with the NSS fit's parameters, loading column 4 of the Svensson loadings, scaled by `strength`;
        present only from month tau on ("break"); absent in "null3"; present throughout in "null4" (size at d=4).
    Noise: N(0, s_i^2) per maturity with s_i the d=3 fit's residual s.d. in every regime (no change in noise at the
        break), optionally AR(1) over time with coefficient `resid_rho` and the same marginal variance.
    Returns y (T, N); with return_signal also the noise-free part (rank 3 before tau and 4 after: a testable fact)."""
    rng = np.random.default_rng(seed)
    lam3, rho3, sd3, resid3 = model[3]
    lam4, rho4, sd4, _ = model[4]
    f = sim.simulate_factors(rho3, sd3, T, rng) @ lam3.T
    f4 = sim.simulate_factors(rho4[3:], sd4[3:], T, rng)[:, 0] * strength
    on = {"null3": np.zeros(T, bool), "null4": np.ones(T, bool), "break": np.arange(T) >= tau}[setup]
    signal = f + np.outer(f4 * on, lam4[:, 3])
    e = rng.standard_normal((T, lam3.shape[0]))
    if resid_rho:
        for t in range(1, T):  # AR(1), stationary start below: scale innovations to keep the marginal variance
            e[t] = resid_rho * e[t - 1] + np.sqrt(1 - resid_rho ** 2) * e[t]
    y = signal + e * (resid3 if resid_from == 3 else model[4][3])  # resid_from=4: noise s.d. of the 4-factor fit (diagnostic)
    return (y, signal) if return_signal else y


# ---------------------------------------------------------------- methods
VERSION = "w9-phase0-2"  # part of every cache key: bump when a method or the DGP changes


class FullSample:
    """One Su-Wang test on the whole panel (no early stop: p* is exact, so size/power can be read at any level).
    B draws, add-one p; at B=200 the add-one rule rejects exactly when the paper's p* does at 1/5/10%."""

    def __init__(self, d=3, B=200, c=1.0):
        self.d, self.B, self.c = d, B, c  # c: bandwidth = c x the Silverman-type h of the full sample (week 9 bandwidth study)

    def __call__(self, y, seed):
        h = None if self.c == 1.0 else self.c * (2.35 / np.sqrt(12)) * y.shape[0] ** -.2 * y.shape[1] ** -.1
        obs, _, _ = run_test(y, self.d, self.B, seed * 10 + self.d, h=h)
        return {"d": self.d, "p": obs["bootstrap_p_add_one"], "exact": True, "breaks": [], "edge": False}

    def config(self):
        return {"name": "full", "d": self.d, "B": self.B, "v": VERSION, **({"c": self.c} if self.c != 1.0 else {})}


class Scan:
    """The sequential break scan of wk5-wk8 (growing window, retreat, restart) at one level and bandwidth rule.
    Early stop at `alpha`: decisions are exact AT that level only (p is a lower bound), so run one level per cell.
    Each window test uses 999 screen + 9,999 confirm draws (B differs from FullSample: say so in tables).
    legacy_seed keeps wk8's constant bootstrap seed (regression tests only): with it every replication reuses the
    same bootstrap normals per window, i.e. replications are dependent. Default: a seed that depends on the replication.
    Returns breaks = dated breaks; segments that reject already at the minimum window and cannot be dated by retreat
    (edge) are NOT dated breaks: they set edge=True (a rejection for size, not a location) and are kept in all_breaks
    (the wk8 definition, which counted them)."""

    def __init__(self, d=3, alpha=0.01, bw_mode="silverman", bw_value=None, min_window=60, restart=60, run_k=1,
                 legacy_seed=False):
        self.d, self.alpha, self.bw_mode, self.bw_value = d, alpha, bw_mode, bw_value
        self.min_window, self.restart, self.run_k, self.legacy_seed = min_window, restart, run_k, legacy_seed

    def __call__(self, y, seed):
        bw = self.bw_value if self.bw_value is not None else (
            round(float(silverman_th(*y.shape)), 1) if self.bw_mode == "full" else None)
        kw = {} if self.legacy_seed else {"seed_base": 40_000_000 + seed * 10 ** 8}
        tester = WindowTester(y, self.d, alpha=self.alpha, bw_mode=self.bw_mode, bw_value=bw, early_stop=True, **kw)
        segs = segmented_scan(len(y), tester, self.min_window, restart_min_window=self.restart, run_k=self.run_k)
        hit = [g for g in segs if g["break_month"] is not None]
        dated = [g["break_month"] for g in hit if not g["rejected_at_min_window"]]
        return {"d": self.d, "p": None, "exact": False, "alpha": self.alpha, "breaks": dated,
                "edge": any(g["rejected_at_min_window"] for g in hit), "all_breaks": [g["break_month"] for g in hit]}

    def config(self):
        return {"name": "scan", "d": self.d, "alpha": self.alpha, "bw_mode": self.bw_mode, "bw_value": self.bw_value,
                "min_window": self.min_window, "restart": self.restart, "run_k": self.run_k,
                "legacy_seed": self.legacy_seed, "screen": 999, "confirm": 9999, "early_stop": True, "v": VERSION}


# ---------------------------------------------------------------- running
def _one(args):
    method, dgp, seed = args
    maker = legacy_panel if dgp["generator"] == "legacy" else panel
    kw = {k: v for k, v in dgp.items() if k not in ("generator", "T", "setup", "tau", "dataset", "model")}
    y = maker(dgp["model"], dgp["T"], dgp["setup"], dgp["tau"], seed, **kw)
    return {"seed": seed, **method(y, seed)}


def run_cell(method, dgp, seed_list, jobs=5, cache=True):
    """Outcomes of `method` on `dgp` for every seed; finished seeds are cached in results/cells/<hash>.jsonl so an
    interrupted run resumes. dgp: dict(generator, setup, T, tau, [strength, resid_rho], model, dataset)."""
    spec = {"method": method.config(), "dgp": {k: v for k, v in dgp.items() if k != "model"}, "v": VERSION}
    key = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    path = W9 / f"results/cells/{key}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if cache and path.exists():
        done = {r["seed"]: r for r in map(json.loads, path.read_text().splitlines())}
    todo = [s for s in seed_list if s not in done]
    if todo:
        with ProcessPoolExecutor(jobs) as pool, path.open("a") as fh:
            for r in pool.map(_one, [(method, dgp, s) for s in todo], chunksize=1):
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                done[r["seed"]] = r
        (path.with_suffix(".spec.json")).write_text(json.dumps(spec, indent=1))
    return [done[s] for s in seed_list]


# ---------------------------------------------------------------- metrics
def _check(dgp, role, outcomes):
    """Size is only defined where H0 holds for the tested rank d (null3 with d=3, null4 with d=4); power and
    location only on the 3 -> 4 break with d=3."""
    d = outcomes[0]["d"]
    if role == "size" and dgp["setup"] != f"null{d}":
        raise ValueError(f"size of a d={d} procedure needs the null{d} DGP, got {dgp['setup']}")
    if role in ("power", "location") and (dgp["setup"] != "break" or d != 3):
        raise ValueError(f"{role} needs the break DGP tested at d=3, got {dgp['setup']} at d={d}")


def _rejects(outcomes, alpha):
    """Per-panel rejection at `alpha`. Exact p (FullSample): p < alpha at any level. Early-stopped scan: only at the
    level it ran at; a scan rejects if it dates a break or hits an unresolved edge."""
    if outcomes[0]["exact"]:
        return np.array([r["p"] < alpha for r in outcomes])
    if any(abs(r["alpha"] - alpha) > 1e-12 for r in outcomes):
        raise ValueError("scan outcomes are early-stopped at their own alpha; evaluate at that alpha only")
    return np.array([bool(r["breaks"]) or r["edge"] for r in outcomes])


def size(outcomes, dgp, alpha):
    """Empirical size: share of null panels on which the procedure rejects (full sample: p* < alpha; scan: dates a
    break or hits an unresolved edge, i.e. the family-wise error of the whole procedure), with the MC s.e."""
    _check(dgp, "size", outcomes)
    rej = _rejects(outcomes, alpha)
    n, rate = len(outcomes), float(rej.mean())
    return {"n": n, "rate": rate, "se": float(np.sqrt(rate * (1 - rate) / n)),
            "mean_breaks": float(np.mean([len(r["breaks"]) for r in outcomes]))}


def power(outcomes, dgp, alpha, slack=12):
    """Full sample: share of break panels rejected. Scan: share with a dated break at or after tau - slack (a break
    dated earlier is a false alarm in the null part of the data, reported separately as `false_alarm_pre`)."""
    _check(dgp, "power", outcomes)
    n, tau = len(outcomes), dgp["tau"]
    if outcomes[0]["exact"]:
        rate = float(_rejects(outcomes, alpha).mean())
        return {"n": n, "rate": rate, "se": float(np.sqrt(rate * (1 - rate) / n))}
    _rejects(outcomes, alpha)  # level check
    rate = float(np.mean([any(b >= tau - slack for b in r["breaks"]) for r in outcomes]))
    return {"n": n, "rate": rate, "se": float(np.sqrt(rate * (1 - rate) / n)),
            "false_alarm_pre": float(np.mean([any(b < tau - slack for b in r["breaks"]) for r in outcomes])),
            "any_break": float(np.mean([bool(r["breaks"]) for r in outcomes]))}


def location(outcomes, dgp, tols=(12, 24, 36)):
    """Dating of scan-type methods; all panels in the denominator, a missed break counts as a miss.
    lag = estimate - tau. A forward-growing scan dates a break at the month whose inclusion first rejects, so lag >= 0
    for a true detection by construction (it is a detection delay, not an error centred on 0); within_k = share of
    panels with |lag| <= k months.
    first: the first dated break is the estimate. nearest: the dated break closest to tau (lenient: rewards
    over-reporting, so read it with mean_extra_breaks)."""
    _check(dgp, "location", outcomes)
    tau, n = dgp["tau"], len(outcomes)
    first = [r["breaks"][0] if r["breaks"] else None for r in outcomes]
    near = [min(r["breaks"], key=lambda b: abs(b - tau)) if r["breaks"] else None for r in outcomes]
    out = {"n": n, "missed": float(np.mean([f is None for f in first]))}
    for name, est in (("first", first), ("nearest", near)):
        lag = np.array([e - tau for e in est if e is not None], float)
        out[name] = {"median_lag": float(np.median(lag)) if len(lag) else None,
                     "mad": float(np.median(np.abs(lag - np.median(lag)))) if len(lag) else None,
                     **{f"within_{t}": float(np.sum(np.abs(lag) <= t) / n) for t in tols}}
    out["early_false"] = float(np.mean([f is not None and f < tau - 12 for f in first]))
    out["mean_extra_breaks"] = float(np.mean([max(len(r["breaks"]) - 1, 0) for r in outcomes]))
    return out
