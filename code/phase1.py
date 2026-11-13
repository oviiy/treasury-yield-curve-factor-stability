"""Phase 1 baseline (JKV, T=480, d=3 unless noted) on the TUNE seeds.

Stage A (cheap): full-sample test on null3 / null4 (size at d=3 / d=4), break at tau = T/3, T/2, 2T/3 (power), plus weaker
  fourth factors (strength 0.5, 0.25; informational, Phase 3 material) -- 500 panels (200 for the weak ones).
Stage B (slow): old scan (Silverman h per window) and new scan (full-sample T*h) at 1%, null3 and break at the three taus.
Usage: python3 scripts/phase1.py A | B [reps] [jobs]; cells are cached in results/cells so reruns resume.
"""
import sys

import benchmark as bm

T = 480
TAUS = {"1/3": 160, "1/2": 240, "2/3": 320}


def D(model, setup, tau=240, **k):
    return {"generator": "w9", "setup": setup, "T": T, "tau": tau, "model": model, "dataset": "jkv", **k}


def stage_a(model, jobs):
    for d, setup in ((3, "null3"), (4, "null4")):
        bm.run_cell(bm.FullSample(d), D(model, setup), bm.seeds("tune", 500), jobs)
        print("A done", setup, flush=True)
    for name, tau in TAUS.items():
        bm.run_cell(bm.FullSample(3), D(model, "break", tau), bm.seeds("tune", 500), jobs)
        for strength in (0.5, 0.25):
            bm.run_cell(bm.FullSample(3), D(model, "break", tau, strength=strength), bm.seeds("tune", 200), jobs)
        print("A done tau", name, flush=True)


def stage_b(model, reps, jobs):
    S = bm.seeds("tune", reps)
    for setup, tau in [("null3", 240)] + [("break", t) for t in TAUS.values()]:
        for bw in ("silverman", "full"):
            bm.run_cell(bm.Scan(3, 0.01, bw), D(model, setup, tau), S, jobs)
            print("B done", setup, tau, bw, flush=True)


if __name__ == "__main__":
    model, _ = bm.load_model("jkv")
    stage = sys.argv[1]
    jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    if stage == "A":
        stage_a(model, jobs)
    else:
        stage_b(model, int(sys.argv[2]) if len(sys.argv) > 2 else 100, jobs)
