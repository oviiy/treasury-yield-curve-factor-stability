"""Fast proof of concept of bootstrap-free dating statistics (week 9, our addition; user request 29 Oct).

Every method computes a curve G(t) from principal-component fits only (no bootstrap), dates the break at argmax G and
reports a break if max G exceeds a critical value. The critical value is the 95% quantile of max G on 200 null3 panels
from the CALIBRATION seeds (tune block); size is measured on 200 other null3 panels (validate block). This "oracle"
calibration uses the known DGP and is not available on real data: it isolates how well each statistic separates and
dates a 3 -> 4 change; a real-data version needs a bootstrap (and inherits the autocorrelation problem).

Methods (X = panel standardised by column; RSS_d(A) = residual sum of squares of a rank-d PCA fit on rows A, no
centring, as in the Su-Wang fits):
  split   full-sample least squares split: G(k) = [RSS(all) - RSS(<k) - RSS(>=k)] / RSS(all), 10% trimmed
          (the dating step of Gated-split; here with its own calibrated threshold instead of the Su-Wang gate).
  mosum   two-sided windows of h months: G(t) = [RSS(L u R) - RSS(L) - RSS(R)] / RSS(L u R), L = [t-h, t), R = [t, t+h).
  eig4    factor count: share of the 4th eigenvalue in R minus that in L (same windows) -- targets "3 -> 4" directly;
          eig4abs: the 4th eigenvalue itself (per month), R minus L; eigratio: log(lambda4/lambda5), R minus L.
  subsp   loading shift: distance between the top-3 eigenvector subspaces of L and R, 3 - tr(P_L P_R).
Usage: python3 scripts/newmethods_poc.py [jobs]  -> results/newmethods_poc.md"""
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import benchmark as bm

D_RANK, T = 3, 480
WINDOWS = (60, 96)


def _eig(xx):
    return np.linalg.eigh(xx)  # ascending


def _rss(xx, d=D_RANK):
    w = np.linalg.eigvalsh(xx)
    return float(w[:-d].sum())


def curves(y):
    x = (y - y.mean(0)) / y.std(0, ddof=1)
    n = x.shape[1]
    outer = np.einsum("ti,tj->tij", x, x)
    cum = np.concatenate([np.zeros((1, n, n)), np.cumsum(outer, 0)])  # cum[a] = sum of rows < a
    S = lambda a, b: cum[b] - cum[a]
    out = {}
    tot = S(0, T)
    rss_all = _rss(tot)
    lo, hi = int(0.1 * T), int(0.9 * T)
    out["split"] = {k: (rss_all - _rss(S(0, k)) - _rss(S(k, T))) / rss_all for k in range(lo, hi + 1)}
    for h in WINDOWS:
        ms, e4, sp, ea, er = {}, {}, {}, {}, {}
        for t in range(h, T - h + 1):
            L, R = S(t - h, t), S(t, t + h)
            pooled = L + R
            rp = _rss(pooled)
            ms[t] = (rp - _rss(L) - _rss(R)) / rp
            wl, vl = _eig(L)
            wr, vr = _eig(R)
            e4[t] = wr[-4] / wr.sum() - wl[-4] / wl.sum()
            ea[t] = (wr[-4] - wl[-4]) / h                          # 4th eigenvalue per month, right minus left
            er[t] = np.log(wr[-4] / wr[-5]) - np.log(wl[-4] / wl[-5])  # eigenvalue ratio 4/5 (Ahn-Horenstein style)
            pl, pr = vl[:, -3:], vr[:, -3:]
            sp[t] = 3 - float(np.sum((pl.T @ pr) ** 2))
        out[f"mosum{h}"], out[f"eig4_{h}"], out[f"subsp{h}"] = ms, e4, sp
        out[f"eig4abs{h}"], out[f"eigratio{h}"] = ea, er
    return out


def _one(args):
    model, setup, tau, seed, kw = args
    y = bm.panel(model, T, setup, tau, seed, **kw)
    return {k: (max(c, key=c.get), max(c.values())) for k, c in curves(y).items()}  # (argmax, max)


def run(model, setup, tau, seeds, jobs, **kw):
    with ProcessPoolExecutor(jobs) as pool:
        return list(pool.map(_one, [(model, setup, tau, s, kw) for s in seeds], chunksize=4))


def main(jobs=5):
    model, _ = bm.load_model("jkv")
    cal = run(model, "null3", 240, bm.seeds("tune", 200), jobs)
    names = list(cal[0])
    crit = {k: float(np.quantile([r[k][1] for r in cal], 0.95)) for k in names}
    cells = {"size": run(model, "null3", 240, bm.seeds("validate", 200), jobs),
             "size AR rho=.5 (iid crit)": run(model, "null3", 240, bm.seeds("validate", 200), jobs, resid_rho=0.5)}
    for st in (1.0, 0.5):
        for tau in (160, 240, 320):
            cells[(st, tau)] = run(model, "break", tau, bm.seeds("tune", 100), jobs, **({} if st == 1 else {"strength": st}))
    cells[("ar", 240)] = run(model, "break", 240, bm.seeds("tune", 100), jobs, resid_rho=0.5)

    def stats(rs, k, tau):
        det = [r[k][1] > crit[k] for r in rs]
        lag = np.array([r[k][0] - tau for r in rs], float)
        found = [d and (r[k][0] >= tau - 12) for d, r in zip(det, rs)]
        within = [d and abs(r[k][0] - tau) <= 24 for d, r in zip(det, rs)]
        return 100 * np.mean(found), float(np.median(lag[det])) if any(det) else None, 100 * np.mean(within), \
            float(np.median(np.abs(lag)))  # last: dating error of argmax regardless of detection

    lines = ["# 新定日期方法 PoC（JKV 校准，d = 3；临界值 = null3 上 max 统计量的 95% 分位数，tune 200 个面板）", "",
             "size 用 validate 种子的 200 个 null3 面板；断点格 100 个 tune 面板。power = 统计量超过临界值且 argmax ≥ τ − 12；"
             "±24 = 超过临界值且 |argmax − τ| ≤ 24；|误差|中位 = 不论是否超过临界值，argmax 与 τ 距离的中位数（纯定位能力）。", "",
             "| 方法 | size % | AR ρ=.5 size % | 强度 | τ | power % | 滞后中位 | ±24 % | |误差|中位 |", "|---" * 9 + "|"]
    summary = {}
    for k in names:
        sz = 100 * np.mean([r[k][1] > crit[k] for r in cells["size"]])
        szar = 100 * np.mean([r[k][1] > crit[k] for r in cells["size AR rho=.5 (iid crit)"]])
        summary[k] = {"size": sz, "size_ar": szar}
        for st in (1.0, 0.5):
            for tau in (160, 240, 320):
                p, lag, w, ae = stats(cells[(st, tau)], k, tau)
                summary[k][(st, tau)] = (p, lag, w, ae)
                lines.append(f"| {k} | {sz:.1f} | {szar:.1f} | {st:g} | {tau} | {p:.0f} | {lag} | {w:.0f} | {ae:.1f} |")
        p, lag, w, ae = stats(cells[("ar", 240)], k, 240)
        summary[k]["ar"] = (p, lag, w, ae)
        lines.append(f"| {k} | {sz:.1f} | {szar:.1f} | 1, AR ρ=.5 | 240 | {p:.0f} | {lag} | {w:.0f} | {ae:.1f} |")
    (bm.W9 / "results/newmethods_poc.md").write_text("\n".join(lines) + "\n")
    np.save(bm.W9 / "results/newmethods_poc_summary.npy", summary, allow_pickle=True)
    print("\n".join(lines))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
