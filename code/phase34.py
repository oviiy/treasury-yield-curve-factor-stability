"""Phase 3 (stress tests, tune seeds) and Phase 4 (final run, FINAL seeds) for the selected method, Gated-split:
the Su-Wang full-sample test decides (alpha), the least-squares split dates. FullSample (exact p) is run alongside.
Usage: python3 scripts/phase34.py stress|final [jobs]. Prints one line per cell; results cached in results/cells."""
import sys

import benchmark as bm
import methods as M
from explore2 import auto_jobs

TAUS = (160, 240, 320)


def D(model, dataset, setup, tau=240, **k):
    return {"generator": "w9", "setup": setup, "T": 480, "tau": tau, "model": model, "dataset": dataset, **k}


def line(tag, m, model, dataset, S, jobs, **kw):
    a = m.alpha
    n = bm.run_cell(m, D(model, dataset, "null3", **kw), S, auto_jobs(jobs))
    sz = bm.size(n, D(model, dataset, "null3", **kw), a)
    out = [f"{tag:44s} size {100*sz['rate']:4.1f}±{100*sz['se']:.1f}"]
    return out


def gated_row(tag, model, dataset, S, jobs, strengths, alpha=0.05, taus=TAUS, **kw):
    m = M.GatedSplit(3, alpha)
    n = bm.run_cell(m, D(model, dataset, "null3", **kw), S, auto_jobs(jobs))
    sz = bm.size(n, D(model, dataset, "null3", **kw), alpha)
    parts = [f"{tag:40s} a={alpha} size {100*sz['rate']:4.1f}±{100*sz['se']:.1f}"]
    for st in strengths:
        cells = []
        for tau in taus:
            dg = D(model, dataset, "break", tau, **({} if st == 1.0 else {"strength": st}), **kw)
            b = bm.run_cell(m, dg, S, auto_jobs(jobs))
            p, l = bm.power(b, dg, alpha), bm.location(b, dg)
            f = l["first"]
            cells.append(f"t{tau} pw {100*p['rate']:.0f} lag {f['median_lag']} mad {f['mad']} in24 {100*f['within_24']:.0f} early {100*l['early_false']:.0f}")
        parts.append(f"| s={st}: " + "; ".join(cells))
    print(" ".join(parts), flush=True)


if __name__ == "__main__":
    stage = sys.argv[1]
    jobs = sys.argv[2] if len(sys.argv) > 2 else "auto"
    jkv, _ = bm.load_model("jkv")
    if stage == "stress":
        S = bm.seeds("tune", 200)
        gated_row("strength grid", jkv, "jkv", S, jobs, (0.3, 0.4, 0.6, 0.75))
        for rho in (0.5, 0.7):
            gated_row(f"AR(1) noise rho={rho}", jkv, "jkv", S, jobs, (1.0, 0.5), taus=(240,), resid_rho=rho)
        gated_row("noise of the 4-factor fit", jkv, "jkv", S, jobs, (1.0, 0.5), resid_from=4)
        tsy, _ = bm.load_model("treasury_cmt")
        gated_row("Treasury CMT calibration (N=8)", tsy, "treasury_cmt", S, jobs, (1.0, 0.5))
    else:
        S = bm.seeds("final", 500)
        for alpha in (0.05, 0.01):
            gated_row("FINAL jkv", jkv, "jkv", S, jobs, (1.0, 0.5), alpha=alpha)
        m = bm.FullSample(3)
        n = bm.run_cell(m, D(jkv, "jkv", "null3"), S, auto_jobs(jobs))
        row = ["FINAL full-sample size " + "/".join(f"{100*bm.size(n, D(jkv, 'jkv', 'null3'), a)['rate']:.1f}" for a in (0.01, 0.05, 0.10))]
        for st in (1.0, 0.5):
            for tau in TAUS:
                dg = D(jkv, "jkv", "break", tau, **({} if st == 1.0 else {"strength": st}))
                b = bm.run_cell(m, dg, S, auto_jobs(jobs))
                row.append(f"s={st} t{tau} power " + "/".join(f"{100*bm.power(b, dg, a)['rate']:.1f}" for a in (0.01, 0.05, 0.10)))
        print(" | ".join(row), flush=True)
