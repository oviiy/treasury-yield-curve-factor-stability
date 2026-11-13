"""Exploratory dating methods and AR(1) bootstrap trials.

These methods are secondary to the full-sample test. Real-data break dates
remain provisional until serial dependence is handled reliably.
"""
import numpy as np
import sw_test
from sw_test import run_test, silverman_th, kernel_matrices

VERSION = "public-consolidated-v1"

def _degenerate(result):
    return result.get("stage") == "degenerate"

def _retreat(seg_start, window, test_window, batch=1, mapper=map):
    """Shrink below `window` until a length does not reject. Returns (break_month or None, trajectory).
    Lengths are tested in batches (in parallel when `mapper` is a pool map); results are used in order."""
    trajectory = []
    ends = [seg_start + length for length in range(window - 1, 0, -1)]
    for i in range(0, len(ends), batch):
        chunk = ends[i:i + batch]
        for e, result in zip(chunk, mapper(test_window, [seg_start] * len(chunk), chunk)):
            trajectory.append({"end": e, "retreat": True, **result})
            if _degenerate(result):
                return None, trajectory
            if not result["reject"]:
                return e, trajectory
    return None, trajectory

def segmented_scan(n_obs, test_window, min_window=100, restart_min_window=None, retreat=True,
                   batch=1, mapper=map, run_k=1):
    """Returns a list of segments: start, break_month (None = stable to the end), rejected_at_min_window (an
    unresolved edge), retreated (break dated by retreat), trajectory (all tested windows, in order).

    run_k (week 6): a break needs run_k consecutive rejecting windows; it is dated at the first of them, exactly as
    with a single rejection (run_k = 1, the original rule). A shorter run of rejections followed by a non-rejection is
    ignored and the search goes on; a run still shorter than run_k when the data end counts as no break."""
    restart_min_window = min_window if restart_min_window is None else restart_min_window
    segments, seg_start, is_first = [], 0, True
    while True:
        window = min_window if is_first else restart_min_window
        if n_obs - seg_start < window:
            break
        trajectory, break_end, first_test, run_start = [], None, None, None
        ends = list(range(seg_start + window, n_obs + 1))
        for i in range(0, len(ends), batch):
            chunk = ends[i:i + batch]
            for e, result in zip(chunk, mapper(test_window, [seg_start] * len(chunk), chunk)):
                trajectory.append({"end": e, **result})
                if first_test is None and not _degenerate(result):
                    first_test = e
                if result["reject"]:
                    run_start = e if run_start is None else run_start
                    if e - run_start + 1 >= run_k:
                        break_end = run_start
                        break
                else:
                    run_start = None
            if break_end is not None:
                break
        edge = break_end is not None and break_end == first_test
        if edge and retreat:
            brk, back = _retreat(seg_start, break_end - seg_start, test_window, batch, mapper)
            if brk is not None:
                # the break test is the window [s, brk + 1): the shortest one that still rejects
                brk_test = back[-2] if len(back) > 1 else next(t for t in trajectory if t["end"] == break_end)
                segments.append({"start": seg_start, "break_month": brk, "rejected_at_min_window": False,
                                 "retreated": True, "break_p": brk_test.get("p"), "trajectory": back[::-1] + trajectory})
                seg_start, is_first = brk, False
                continue
            trajectory = back[::-1] + trajectory
        brk_p = None if break_end is None else next(t for t in trajectory if t["end"] == break_end and not t.get("retreat")).get("p")
        segments.append({"start": seg_start, "break_month": None if break_end is None else break_end - 1,
                         "rejected_at_min_window": edge, "retreated": False,
                         "break_p": brk_p, "trajectory": trajectory})
        if break_end is None:
            break
        seg_start, is_first = break_end - 1, False
    return segments

class WindowTester:
    """Picklable tester: 999-draw screen, then 9,999-draw confirm on screen rejections."""

    def __init__(self, y, rank, screen_reps=999, confirm_reps=9999, seed_base=40_000_000, alpha=0.01,
                 bw_mode="silverman", bw_value=None, early_stop=False):
        self.y, self.rank, self.alpha = y, rank, alpha
        self.early_stop = early_stop  # week 7: same decisions, far fewer draws; p / critical_99 not exact (simulations only)
        self.bw_mode, self.bw_value = bw_mode, bw_value
        self.screen_reps, self.confirm_reps, self.seed_base = screen_reps, confirm_reps, seed_base

    def bandwidth(self, t):
        """h for a window of t months. silverman: the rule, recomputed from t (None = kernel_matrices' own rule);
        fixed_h: the same h in every window; fixed_th: the same effective bandwidth T*h in months (h capped at 1; NOT
        valid when h > 0.5, see "scale"); scale: bw_value x the Silverman h of the window (h capped at 0.5)."""
        if self.bw_mode == "full":
            # week 8 (advisor, 2026-10-21): the full-sample effective bandwidth T*h (months) in every window, not capped;
            # h > 1/2 (even > 1) is handled by the renormalised boundary kernel in wk4_common.kernel_matrices
            return self.bw_value / t
        if self.bw_mode == "fixed_h":
            return self.bw_value
        if self.bw_mode == "fixed_th":
            return min(self.bw_value / t, 1.0)
        if self.bw_mode == "scale":
            # week 7 (revised): c x Silverman's h for this window, capped at 0.5 -- above 0.5 the two edge zones of the
            # boundary correction in su_wang_monthly.kernel_matrices overlap and the kernel weights no longer sum to one
            return min(self.bw_value * (2.35 / np.sqrt(12)) * t ** -.2 * self.y.shape[1] ** -.1, 0.5)
        return None

    def __call__(self, s, e):
        seed = self.seed_base + self.rank * 10**7 + s * 10**4 + e
        try:
            obs, _, _ = run_test(self.y[s:e], self.rank, self.screen_reps, seed, h=self.bandwidth(e - s),
                                   stop_alpha=self.alpha if self.early_stop else None)
        except ValueError:
            # Window too short for local PCA at this rank: no evidence, so no rejection.
            return {"J": float("nan"), "critical_99": float("nan"), "p": float("nan"), "reject": False,
                    "stage": "degenerate", "seed": seed, "Th_months": float("nan")}
        out = {"J": obs["J"], "critical_99": obs["critical_99"], "p": obs["bootstrap_p_add_one"],
               "reject": obs["bootstrap_p_add_one"] < self.alpha, "stage": "screen", "seed": seed, "Th_months": obs["Th_months"]}
        if out["reject"] and self.confirm_reps:
            cseed = seed + 5 * 10**6
            c, _, _ = run_test(self.y[s:e], self.rank, self.confirm_reps, cseed, h=self.bandwidth(e - s),
                           stop_alpha=self.alpha if self.early_stop else None)
            out.update(J=c["J"], critical_99=c["critical_99"], p=c["bootstrap_p_add_one"],
                       reject=c["bootstrap_p_add_one"] < self.alpha, stage="confirm", seed=cseed)
        return out

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

def run_test_ar1(y, rank, reps, seed, stop_alpha, resid="global"):
    """Su-Wang test with an AR(1)-sieve bootstrap (week 9 candidate; NOT the paper's bootstrap). The paper draws the
    bootstrap noise independently over time; real JKV residuals have lag-1 autocorrelation 0.4-0.6, and with AR(1) noise
    the paper's test rejects every null panel. Here each maturity's residual gets its own AR(1) coefficient rho_i
    (lag-1 autocorrelation of the fixed-loading residual, clipped to [0, 0.95]); the innovations keep the paper's
    covariance (x 0.99^|i-j|). Same statistic, kernel and bandwidth; early stop at stop_alpha. Returns the add-one p."""
    x = (y - y.mean(0)) / y.std(0, ddof=1)
    t, n = x.shape
    h, k, conv = kernel_matrices(t, n, None)
    fitted = sw_test.fit(x, rank, k)
    obs = sw_test.statistic(x, fitted, h, k, conv)
    # resid="global" (first version): rho and innovations from the fixed-loading residual, rho clipped to [0, 0.95]. Under a
    # break that residual still holds the missed 4th factor (persistent), so rho is overstated and power is lost
    # mechanically (audit 2026-10-29). resid="local": from the kernel-local fit's residual, rho clipped to [-0.95, 0.95].
    e0 = x - fitted["fixed_common" if resid == "global" else "common"]
    lo = 0 if resid == "global" else -0.95
    rho = np.clip([np.corrcoef(e0[1:, i], e0[:-1, i])[0, 1] for i in range(n)], lo, 0.95)
    u = e0[1:] - rho * e0[:-1]
    cov = (u.T @ u / (t - 1)) * .99 ** np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    eig, vec = np.linalg.eigh(cov)
    root = vec @ np.diag(np.sqrt(np.maximum(eig, 0)))
    rng = np.random.default_rng(seed)
    exceed = 0
    for b in range(reps):
        z = rng.standard_normal((t + 100, n)) @ root.T
        e = np.zeros_like(z)
        for i in range(1, t + 100):
            e[i] = rho * e[i - 1] + z[i]
        xb = fitted["fixed_common"] + e[100:]
        exceed += sw_test.statistic(xb, sw_test.fit(xb, rank, k), h, k, conv)["J"] >= obs["J"]
        if (exceed + 1) / (reps + 1) >= stop_alpha:
            break
    return (exceed + 1) / (reps + 1)

def _sse(x, r):
    s = np.linalg.svd(x - x.mean(0), compute_uv=False)
    return float((s[r:] ** 2).sum())

def _std(y):
    return (y - y.mean(0)) / y.std(0, ddof=1)

def _msr_given(A, B_, d):
    """Mean squared residual of B_ after projecting off the top-d principal directions of A (centred at A's mean)."""
    _, _, vt = np.linalg.svd(A - A.mean(0), full_matrices=False)
    v = vt[:d].T
    z = B_ - A.mean(0)
    return float(((z - z @ v @ v.T) ** 2).mean())

def _eig_ratio(x, d):
    """lambda_{d+1} over the mean of the smaller eigenvalues of the segment's covariance: large when a (d+1)-th factor
    is present in the segment (an eigenvalue-ratio idea in the spirit of Ahn-Horenstein 2013)."""
    s = np.linalg.svd(x - x.mean(0), compute_uv=False) ** 2
    return s[d] / s[d + 1:].mean()

def split_stat(x, d, taus, kind):
    """Statistic for every candidate split. sym / asym: gain of two separate fits over the pooled d-factor fit (both
    sides rank d / one side rank d+1). proj: do the d directions of one side explain the other side as well as they
    explain their own side? max of the two ratios of mean squared residuals (a break makes the foreign side fit worse)."""
    pool = _sse(x, d) if kind in ("sym", "asym") else 0.0
    out = np.empty(len(taus))
    if kind == "asympen":
        # least squares for "d factors on one side, d+1 on the other", with the noise an extra component absorbs in a
        # pure-noise segment subtracted (largest eigenvalue of an n x (N-d) noise matrix ~ sigma^2 (sqrt n + sqrt(N-d))^2,
        # Marchenko-Pastur edge); sigma^2 from the pooled (d+1)-factor fit. Returns minus the criterion (argmax = date).
        T, N = x.shape
        sig2 = _sse(x, d + 1) / (T * (N - d - 1))
        pen = lambda n: sig2 * (np.sqrt(n) + np.sqrt(N - d)) ** 2
        for i, t in enumerate(taus):
            L, R = x[:t], x[t:]
            out[i] = -min(_sse(L, d) + _sse(R, d + 1) + pen(len(R)), _sse(L, d + 1) + _sse(R, d) + pen(len(L)))
        return out
    for i, t in enumerate(taus):
        L, R = x[:t], x[t:]
        if kind == "eig":
            out[i] = abs(np.log(_eig_ratio(R, d)) - np.log(_eig_ratio(L, d)))
            continue
        if kind == "proj":
            own_l, own_r = _sse(L, d) / L.size, _sse(R, d) / R.size
            out[i] = max(_msr_given(L, R, d) / own_l, _msr_given(R, L, d) / own_r)
            continue
        if kind == "sym":
            best = _sse(L, d) + _sse(R, d)
        else:
            best = min(_sse(L, d) + _sse(R, d + 1), _sse(L, d + 1) + _sse(R, d))
        out[i] = pool - best
    return out

def _grid(T, step, trim):
    lo = max(int(round(trim * T)), 36)
    return np.arange(lo, T - lo + 1, step)

class SupSplit:
    """recursive=True: binary segmentation, i.e. the same test on each side of a dated break (segments of at least
    2 x the trimmed minimum), so the method can report several breaks like the scan."""

    def __init__(self, d=3, alpha=0.05, B=199, step=6, trim=0.1, kind="sym", recursive=False, boot="paper"):
        self.d, self.alpha, self.B, self.step, self.trim, self.kind, self.recursive = d, alpha, B, step, trim, kind, recursive
        self.boot = boot  # paper: residual covariance x 0.99^|i-j| (as the Su-Wang bootstrap); diag: see _test

    def date(self, x):
        taus = _grid(len(x), self.step, self.trim)
        g = split_stat(x, self.d, taus, self.kind)
        t0 = int(taus[np.argmax(g)])
        lo = max(int(round(self.trim * len(x))), 36)
        fine = np.arange(max(t0 - self.step, lo), min(t0 + self.step, len(x) - lo) + 1)
        gf = split_stat(x, self.d, fine, self.kind)
        return int(fine[np.argmax(gf)]), float(gf.max())

    def _test(self, x, seed):
        T, N = x.shape
        taus = _grid(T, self.step, self.trim)
        if len(taus) < 2:
            return 1.0
        obs = split_stat(x, self.d, taus, self.kind).max()
        u, s, vt = np.linalg.svd(x - x.mean(0), full_matrices=False)
        fitted = x.mean(0) + (u[:, :self.d] * s[:self.d]) @ vt[:self.d]
        e0 = x - fitted
        if getattr(self, "boot", "paper") == "diag":
            # PCA residuals are orthogonal to the fitted loadings, so a bootstrap with their covariance has (almost) no noise
            # in the loading directions and is too quiet overall; draw independent noise with each maturity's residual
            # variance, corrected for the d fitted dimensions: / ((1 - d/N)(1 - d/T))
            root = np.diag(np.sqrt((e0 ** 2).mean(0) / ((1 - self.d / N) * (1 - self.d / T))))
        else:
            cov = (e0.T @ e0 / T) * .99 ** np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])
            eig, vec = np.linalg.eigh(cov)
            root = vec @ np.diag(np.sqrt(np.maximum(eig, 0)))
        rng = np.random.default_rng(seed)
        exceed = 0
        for b in range(self.B):
            xb = fitted + rng.standard_normal((T, N)) @ root.T
            exceed += split_stat(xb, self.d, taus, self.kind).max() >= obs
            if (exceed + 1) / (self.B + 1) >= self.alpha:
                break
        return (exceed + 1) / (self.B + 1)

    def _segment(self, x, offset, seed, depth, out):
        p = self._test(x, seed)
        if p >= self.alpha:
            return p
        t, _ = self.date(x)
        out.append(offset + t)
        if self.recursive and depth < 3:
            for part, off in ((x[:t], offset), (x[t:], offset + t)):
                if len(part) >= 4 * max(int(round(self.trim * len(part))), 36) // 2 and len(part) >= 2 * 72:
                    self._segment(part, off, seed * 7 + off + 1, depth + 1, out)
        return p

    def __call__(self, y, seed):
        x = _std(y)
        out = []
        p = self._segment(x, 0, seed * 10 + self.d, 0, out)
        return {"d": self.d, "p": p, "exact": False, "alpha": self.alpha, "edge": False, "breaks": sorted(out)}

    def config(self):
        return {"name": "supsplit", "d": self.d, "alpha": self.alpha, "B": self.B, "step": self.step, "trim": self.trim,
                "kind": self.kind, "recursive": self.recursive, "v": "m2", **({"boot": self.boot} if getattr(self, "boot", "paper") != "paper" else {})}

class GatedSplit:
    """Su-Wang full-sample test decides; the least-squares split (sym) dates. recursive=True: binary segmentation, the
    same test (on the segment, its own bandwidth) on each side of a dated break while the segment has >= min_seg months."""

    def __init__(self, d=3, alpha=0.05, B=200, step=6, trim=0.1, kind="sym", recursive=False, min_seg=120, boot="paper"):
        self.d, self.alpha, self.B, self.recursive, self.min_seg, self.boot = d, alpha, B, recursive, min_seg, boot
        self.loc = SupSplit(d, alpha, B, step, trim, kind)

    def _seg(self, y, offset, seed, depth, out):
        if getattr(self, "boot", "paper") in ("ar1", "ar1loc"):
            p = run_test_ar1(y, self.d, self.B, seed, self.alpha, resid="local" if self.boot == "ar1loc" else "global")
        else:
            obs, _, _ = run_test(y, self.d, self.B, seed, stop_alpha=self.alpha)
            p = obs["bootstrap_p_add_one"]
        if p >= self.alpha:
            return p
        t = self.loc.date(_std(y))[0]
        out.append(offset + t)
        if getattr(self, "recursive", False) and depth < 3:
            for part, off in ((y[:t], offset), (y[t:], offset + t)):
                if len(part) >= self.min_seg:
                    self._seg(part, off, seed * 7 + off + 1, depth + 1, out)
        return p

    def __call__(self, y, seed):
        out = []
        p = self._seg(y, 0, seed * 10 + self.d, 0, out)
        return {"d": self.d, "p": p, "exact": False, "alpha": self.alpha, "edge": False, "breaks": sorted(out)}

    def config(self):
        c = {"name": "gatedsplit", "d": self.d, "alpha": self.alpha, "B": self.B, "step": self.loc.step,
             "trim": self.loc.trim, "kind": self.loc.kind, "v": "m2"}
        if getattr(self, "recursive", False):
            c.update(recursive=True, min_seg=self.min_seg)
        if getattr(self, "boot", "paper") != "paper":
            c.update(boot=self.boot)
        return c

class Scan2:
    def __init__(self, d=3, alpha=0.01, bw_mode="full", min_window=60, restart=60, step=6, B=999, gate=None, bw_value=None):
        self.d, self.alpha, self.bw_mode, self.min_window, self.restart = d, alpha, bw_mode, min_window, restart
        self.bw_value = bw_value  # for bw_mode "scale": c x the window's own Silverman h (capped at 1/2)
        self.step, self.B, self.gate = step, B, gate

    def __call__(self, y, seed):
        T = len(y)
        if self.gate is not None:
            obs, _, _ = run_test(y, self.d, 200, seed * 10 + self.d, stop_alpha=self.gate)
            if obs["bootstrap_p_add_one"] >= self.gate:
                return {"d": self.d, "p": None, "exact": False, "alpha": self.alpha, "breaks": [], "edge": False}
        bw = round(float(silverman_th(*y.shape)), 1) if self.bw_mode == "full" else self.bw_value
        tester = WindowTester(y, self.d, screen_reps=self.B, confirm_reps=0, alpha=self.alpha, bw_mode=self.bw_mode,
                              bw_value=bw, early_stop=True, seed_base=40_000_000 + seed * 10 ** 8)
        breaks, edge, s, win = [], False, 0, self.min_window
        while T - s >= win:
            ends = list(range(s + win, T + 1, self.step))
            if ends and ends[-1] != T:
                ends.append(T)
            prev, found, first = None, None, None
            for e in ends:
                r = tester(s, e)
                if first is None and r["stage"] != "degenerate":
                    first = e
                if r["reject"]:
                    if e == first:                          # rejects at the very first testable window: no date
                        edge, found = True, ("edge", e - 1)
                    else:
                        lo = (prev if prev is not None else e - self.step) + 1
                        month = next((m for m in range(lo, e) if tester(s, m)["reject"]), e)
                        found = ("break", month - 1)
                    break
                prev = e
            if found is None:
                break
            if found[0] == "break":
                breaks.append(found[1])
            s, win = found[1], self.restart
        return {"d": self.d, "p": None, "exact": False, "alpha": self.alpha, "breaks": breaks, "edge": edge}

    def config(self):
        return {"name": "scan2", "d": self.d, "alpha": self.alpha, "bw_mode": self.bw_mode, "min_window": self.min_window,
                "restart": self.restart, "step": self.step, "B": self.B, "gate": self.gate, "v": "m1",
                **({"bw_value": self.bw_value} if self.bw_value is not None else {})}

class Rolling:
    def __init__(self, d=3, alpha=0.01, w=120, step=6, B=999):
        self.d, self.alpha, self.w, self.step, self.B = d, alpha, w, step, B

    def __call__(self, y, seed):
        T = len(y)
        bw = round(float(silverman_th(T, y.shape[1])), 1)
        tester = WindowTester(y, self.d, screen_reps=self.B, confirm_reps=0, alpha=self.alpha, bw_mode="full", bw_value=bw,
                              early_stop=True, seed_base=40_000_000 + seed * 10 ** 8)
        for e in range(self.w, T + 1, self.step):
            if tester(e - self.w, e)["reject"]:
                return {"d": self.d, "p": None, "exact": False, "alpha": self.alpha, "edge": False,
                        "breaks": [e - self.w // 2]}
        return {"d": self.d, "p": None, "exact": False, "alpha": self.alpha, "breaks": [], "edge": False}

    def config(self):
        return {"name": "rolling", "d": self.d, "alpha": self.alpha, "w": self.w, "step": self.step, "B": self.B, "v": "m1"}
