"""Week-9 candidate break-detection methods (Phase 2). Same interface as benchmark.Scan / FullSample:
`m(y, seed) -> {"d", "p", "exact": False, "alpha", "breaks", "edge"}`; every method runs at ONE level `alpha`
(early-stopped bootstraps are exact only at that level). `breaks` are dated breaks (months, same convention as the scan:
index of the first post-break month); empty if the procedure does not reject.

  SupSplit   (D+E)  estimation-based: for every candidate split compare the pooled d-factor fit with separate fits on the two
                    sides; reject if the largest gain exceeds its bootstrap critical value; date = the argmax split.
                    kind="sym": left and right both rank d. kind="asym": one side rank d, the other d+1 (the 3 -> 4 alternative).
  GatedSplit (C+D)  full-sample Su-Wang test decides; the split statistic (no bootstrap) only dates.
  Scan2      (A,B)  the wk8 scan on a step grid (tests every `step` months, then refines month by month), any bandwidth rule,
                    optional gate (full-sample test first) and per-window level alpha_w (calibration by Sidak/Bonferroni).
  Rolling    (F)    fixed-length windows every `step` months; first rejecting window end minus w/2 is the date.
"""
import numpy as np

from benchmark import run_test, silverman_th, WindowTester  # noqa: F401  (benchmark pins BLAS threads first)
from wk4_common import kernel_matrices, sw  # noqa: E402


def run_test_ar1(y, rank, reps, seed, stop_alpha, resid="global"):
    """Su-Wang test with an AR(1)-sieve bootstrap (week 9 candidate; NOT the paper's bootstrap). The paper draws the
    bootstrap noise independently over time; real JKV residuals have lag-1 autocorrelation 0.4-0.6, and with AR(1) noise
    the paper's test rejects every null panel. Here each maturity's residual gets its own AR(1) coefficient rho_i
    (lag-1 autocorrelation of the fixed-loading residual, clipped to [0, 0.95]); the innovations keep the paper's
    covariance (x 0.99^|i-j|). Same statistic, kernel and bandwidth; early stop at stop_alpha. Returns the add-one p."""
    x = (y - y.mean(0)) / y.std(0, ddof=1)
    t, n = x.shape
    h, k, conv = kernel_matrices(t, n, None)
    fitted = sw.fit(x, rank, k)
    obs = sw.statistic(x, fitted, h, k, conv)
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
        exceed += sw.statistic(xb, sw.fit(xb, rank, k), h, k, conv)["J"] >= obs["J"]
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
