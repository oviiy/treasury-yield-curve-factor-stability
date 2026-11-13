"""Scan size under AR(1) noise (tune seeds, 100 null panels, rho in {0.5}): the arboot group showed the paper's iid
bootstrap gives 100% size for the full-sample test once the noise is autocorrelated; this checks whether the step-6
scan (same bootstrap inside every window) has the same problem. Usage: python3 scripts/scan_ar.py [jobs] [reps]"""
import sys
import benchmark as bm
import methods as M
from phase1 import D

if __name__ == "__main__":
    jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    model, _ = bm.load_model("jkv")
    S = bm.seeds("tune", reps)
    m = M.Scan2(3, 0.01, step=6)
    for rho in (0.5,):
        dg = D(model, "null3", resid_rho=rho)
        n = bm.run_cell(m, dg, S, jobs)
        sz = bm.size(n, dg, 0.01)
        print(f"scan2 step6 a=.01 full, AR rho={rho}: size {100*sz['rate']:.1f}±{100*sz['se']:.1f} "
              f"brk/null {sz['mean_breaks']:.2f}", flush=True)
