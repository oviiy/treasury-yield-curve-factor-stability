"""Scan calibration (week 9, our addition; pre-registered rule in docs/SCAN_CALIBRATION_PLAN.md).

One pass per panel evaluates a grid of scan rules that share window tests: first/restart window W in {36, 60, 120},
retreat on/off, per-window level alpha_w in {1%, 0.5%, 0.2%}. Every window test (s, e) uses the same bootstrap seed in
every variant and is run once, early-stopped at the loosest level (1%): a window that does not reject at 1% does not
reject at any smaller level, and a window that rejects at 1% ran all B = 999 draws, so its add-one p is exact. Hence
decisions at every alpha_w <= 1% are exact. Scan logic = methods.Scan2 (step 6, then month-by-month refinement; full-sample
T*h bandwidth); retreat: when the very first window rejects, shrink it a month at a time until it no longer rejects
(as the wk5 scan) instead of recording an undated edge.

Usage: python3 scripts/scan_calib.py poc|tune|final [jobs]"""
import sys

import benchmark as bm
from benchmark import silverman_th, WindowTester

WINDOWS, ALPHAS, STEP = (36, 60, 120), (0.01, 0.005, 0.002), 6


def vkey(w, retreat, a):
    return f"W{w}_{'ret' if retreat else 'edge'}_a{a:g}"


class ScanGrid:
    def __init__(self, d=3, B=999):
        self.d, self.B = d, B

    def config(self):
        return {"name": "scangrid", "d": self.d, "B": self.B, "windows": WINDOWS, "alphas": ALPHAS, "step": STEP,
                "bw": "full", "v": 1}

    def __call__(self, y, seed):
        T = len(y)
        tester = WindowTester(y, self.d, screen_reps=self.B, confirm_reps=0, alpha=max(ALPHAS), bw_mode="full",
                              bw_value=round(float(silverman_th(*y.shape)), 1), early_stop=True,
                              seed_base=40_000_000 + seed * 10 ** 8)
        cache = {}

        def p(s, e):
            if (s, e) not in cache:
                r = tester(s, e)
                cache[s, e] = None if r["stage"] == "degenerate" else r["p"]
            return cache[s, e]

        out = {}
        for w in WINDOWS:
            for retreat in (False, True):
                for a in ALPHAS:
                    out[vkey(w, retreat, a)] = self.scan(T, p, w, retreat, a)
        return {"d": self.d, "p": None, "exact": False, "alpha": max(ALPHAS), "breaks": [], "edge": False,
                "variants": out, "windows_tested": len(cache)}

    @staticmethod
    def scan(T, p, win0, retreat, a):
        rej = lambda s, e: (lambda q: q is not None and q < a)(p(s, e))
        breaks, edge, s, win = [], False, 0, win0
        while T - s >= win:
            ends = list(range(s + win, T + 1, STEP))
            if ends and ends[-1] != T:
                ends.append(T)
            prev, found, first = None, None, None
            for e in ends:
                if first is None and p(s, e) is not None:
                    first = e
                if rej(s, e):
                    if e == first:
                        if retreat:
                            m = next((m for m in range(e - 1, s, -1) if p(s, m) is None or not rej(s, m)), None)
                            if m is not None and p(s, m) is not None:
                                found = ("break", m)   # [s, m) is the longest stable stretch: break at month m
                                break
                        edge, found = True, ("edge", e - 1)
                    else:
                        lo = (prev if prev is not None else e - STEP) + 1
                        month = next((m for m in range(lo, e) if rej(s, m)), e)
                        found = ("break", month - 1)
                    break
                prev = e
            if found is None:
                break
            if found[0] == "break":
                breaks.append(found[1])
            s, win = found[1], win0
        return {"breaks": breaks, "edge": edge}


def variant(outcomes, key):
    """Outcomes of one grid variant in the usual {breaks, edge} format (for bm.size / power / location)."""
    return [{**r, "breaks": r["variants"][key]["breaks"], "edge": r["variants"][key]["edge"]} for r in outcomes]


def D(model, setup, tau=240, **k):
    return {"generator": "w9", "setup": setup, "T": 480, "tau": tau, "model": model, "dataset": "jkv", **k}


CELLS = {  # block -> [(setup, tau, strength, n panels)]
    # fast proof of concept (user, 29 Oct: within 1 hour): tune seeds only, small n, no final validation
    "poc": [("null3", 240, 1.0, 100), ("break", 240, 1.0, 30), ("break", 240, 0.5, 30)],
    "tune": [("null3", 240, 1.0, 200), ("break", 160, 1.0, 100), ("break", 240, 1.0, 100), ("break", 320, 1.0, 100),
             ("break", 240, 0.5, 100)],
    "final": [("null3", 240, 1.0, 300), ("break", 160, 1.0, 100), ("break", 240, 1.0, 100), ("break", 320, 1.0, 100),
              ("break", 240, 0.5, 100)],
}

if __name__ == "__main__":
    block = sys.argv[1] if len(sys.argv) > 1 else "tune"
    seed_block = "tune" if block == "poc" else block
    jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    model, _ = bm.load_model("jkv")
    for setup, tau, st, n in CELLS[block]:
        dg = D(model, setup, tau, **({} if st == 1.0 else {"strength": st}))
        bm.run_cell(ScanGrid(), dg, bm.seeds(seed_block, n), jobs)
        print(f"done {block} {setup} tau={tau} s={st} n={n}", flush=True)


def summary(block="poc"):
    """Table of every grid variant on the cells of `block` (cached only; missing cells -> None). Returns rows sorted by
    the pre-registered rule: size <= 5% first, then +-24 at strength 1 (tau = 240), then +-24 at strength 0.5."""
    import final_tables as FT
    model, _ = bm.load_model("jkv")
    seeds_of = lambda n: bm.seeds("tune" if block == "poc" else block, n)
    cells = {(setup, tau, st): FT.get(ScanGrid(), D(model, setup, tau, **({} if st == 1.0 else {"strength": st})), seeds_of(n))
             for setup, tau, st, n in CELLS[block]}
    rows = []
    for w in WINDOWS:
        for retreat in (False, True):
            for a in ALPHAS:
                k = vkey(w, retreat, a)
                n0 = cells[("null3", 240, 1.0)]
                r = {"variant": k, "W": w, "retreat": retreat, "alpha_w": a,
                     "size": 100 * bm.size(variant(n0, k), D(model, "null3"), max(ALPHAS))["rate"] if n0 else None}
                for (setup, tau, st), o in cells.items():
                    if setup == "null3" or o is None:
                        continue
                    dg = D(model, "break", tau, **({} if st == 1.0 else {"strength": st}))
                    l = bm.location(variant(o, k), dg)
                    r[f"power s={st:g} t={tau}"] = 100 * bm.power(variant(o, k), dg, max(ALPHAS))["rate"]
                    r[f"lag s={st:g} t={tau}"] = l["first"]["median_lag"]
                    r[f"in24 s={st:g} t={tau}"] = 100 * l["first"]["within_24"]
                rows.append(r)
    return rows
