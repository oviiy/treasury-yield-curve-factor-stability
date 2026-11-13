"""Phase 2 exploration queue (tune seeds). Each config: null3 (size), break tau in {160,240,320} at strength {1, 0.5}.
Usage: python3 scripts/explore2.py GROUP [jobs] [reps]; GROUP: split | gated | rolling | scan2. Results cached; print one line per config."""
import sys
import benchmark as bm
import methods as M
from phase1 import D

TAUS = (160, 240, 320)


import subprocess


def auto_jobs(jobs):
    """8 cores in total: 3 while the phase-1 scan baseline (5 workers) still runs, 8 after it."""
    if jobs != "auto":
        return int(jobs)
    busy = subprocess.run(["pgrep", "-f", "phase1.py B"], capture_output=True).stdout.strip()
    return 3 if busy else 8


def evaluate(name, m, model, S, jobs, strengths=(1.0, 0.5), taus=TAUS):
    alpha = m.alpha
    n = bm.run_cell(m, D(model, "null3"), S, auto_jobs(jobs))
    sz = bm.size(n, D(model, "null3"), alpha)
    parts = [f"{name:34s} size {100*sz['rate']:4.1f}±{100*sz['se']:.1f} brk/null {sz['mean_breaks']:.2f}"]
    for st in strengths:
        pw, lag, w24, ef, ex = [], [], [], [], []
        for tau in taus:
            kw = {} if st == 1.0 else {"strength": st}
            dg = D(model, "break", tau, **kw)
            b = bm.run_cell(m, dg, S, auto_jobs(jobs))
            pw.append(bm.power(b, dg, alpha)["rate"])
            l = bm.location(b, dg)
            lag.append(l["first"]["median_lag"]); w24.append(l["first"]["within_24"]); ef.append(l["early_false"]); ex.append(l["mean_extra_breaks"])
        f = lambda v: "/".join("-" if x is None else f"{x:.0f}" for x in v)
        parts.append(f"| s={st}: power {f([100*x for x in pw])} lag {f(lag)} in24 {f([100*x for x in w24])} early {f([100*x for x in ef])} extra {sum(ex)/len(ex):.2f}")
    print(" ".join(parts), flush=True)


if __name__ == "__main__":
    group = sys.argv[1]
    jobs = sys.argv[2] if len(sys.argv) > 2 else "auto"
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    model, _ = bm.load_model("jkv")
    S = bm.seeds("tune", reps)
    if group == "split":
        for alpha in (0.05, 0.01, 0.10):
            for trim in (0.1, 0.15, 0.2):
                evaluate(f"sup sym a={alpha} trim={trim} step6", M.SupSplit(3, alpha, trim=trim), model, S, jobs)
        evaluate("sup sym a=.05 trim=.1 step3", M.SupSplit(3, 0.05, step=3), model, S, jobs)
        evaluate("sup sym a=.05 trim=.1 recursive", M.SupSplit(3, 0.05, recursive=True), model, S, jobs)
    elif group == "splitboot":
        for alpha in (0.05, 0.01, 0.10):
            evaluate(f"sup sym a={alpha} boot=diag", M.SupSplit(3, alpha, boot="diag"), model, S, jobs)
    elif group == "gated":
        for alpha in (0.05, 0.01, 0.10):
            evaluate(f"gated split a={alpha}", M.GatedSplit(3, alpha), model, S, jobs)
    elif group == "gatedrec":
        evaluate("gated split a=0.05 recursive", M.GatedSplit(3, 0.05, recursive=True), model, S, jobs)
    elif group == "fullbw":
        # advisor: how does the bandwidth move size, power (and, for the scan, dating)? full-sample test, h = c x rule
        for c in (0.5, 0.75, 1.0, 1.5, 2.0):
            m = bm.FullSample(3, c=c)
            n = bm.run_cell(m, D(model, "null3"), S, auto_jobs(jobs))
            row = [f"full c={c}: size " + "/".join(f"{100*bm.size(n, D(model, 'null3'), a)['rate']:.1f}" for a in (0.01, 0.05, 0.10))]
            for st in (1.0, 0.5):
                dg = D(model, "break", 240, **({} if st == 1.0 else {"strength": st}))
                b = bm.run_cell(m, dg, S, auto_jobs(jobs))
                row.append(f"power s={st} " + "/".join(f"{100*bm.power(b, dg, a)['rate']:.1f}" for a in (0.01, 0.05, 0.10)))
            print(" | ".join(row), flush=True)
    elif group == "rolling":
        for w in (96, 120, 180):
            evaluate(f"rolling w={w} a=.01 step6", M.Rolling(3, 0.01, w), model, S, jobs)
    elif group == "scan2bw":
        # bandwidth study for the scan (step-6 version, 1%): own Silverman h, full-sample T*h, 0.5x and 2x Silverman
        for name, kw in (("full", {"bw_mode": "full"}), ("silverman", {"bw_mode": "silverman"}),
                         ("0.5x", {"bw_mode": "scale", "bw_value": 0.5}), ("2x", {"bw_mode": "scale", "bw_value": 2.0})):
            evaluate(f"scan2 step6 a=.01 bw={name}", M.Scan2(3, 0.01, step=6, **kw), model, S, jobs, taus=(240,))
    elif group == "scan2":
        # step-6 scan, full-sample T*h, 1%, all seven cells: validates the step-6 shortcut against the month-by-month
        # baseline on the same seeds. The GATED scan is derived from these cells plus the FullSample cells (same data, same
        # bootstrap seed: the gate decision equals FullSample p < gate), so it costs nothing extra (see final_tables.py)
        evaluate("scan2 step6 a=.01 full", M.Scan2(3, 0.01, step=6), model, S, jobs)
    elif group == "splitrest":
        # SupSplit with its own bootstrap at the other levels and settings (documents whether ANY setting is size-valid)
        for alpha in (0.01, 0.10):
            evaluate(f"sup sym a={alpha} trim=0.1 step6", M.SupSplit(3, alpha), model, S, jobs)
        for trim in (0.15, 0.2):
            evaluate(f"sup sym a=0.05 trim={trim} step6", M.SupSplit(3, 0.05, trim=trim), model, S, jobs)
        evaluate("sup sym a=.05 trim=.1 recursive", M.SupSplit(3, 0.05, recursive=True), model, S, jobs)
    elif group in ("arboot", "arbootloc"):  # arbootloc: AR(1) bootstrap from the local-fit residual (audit fix)
        # residual autocorrelation: the paper's bootstrap vs an AR(1)-sieve bootstrap, noise rho in {0, .3, .5, .7}
        boots = ("ar1loc",) if group == "arbootloc" else ("paper", "ar1")
        for rho in (0.0, 0.3, 0.5, 0.7):
            kw = {} if rho == 0 else {"resid_rho": rho}
            for boot in boots:
                m = M.GatedSplit(3, 0.05, boot=boot)
                n = bm.run_cell(m, D(model, "null3", **kw), S, auto_jobs(jobs))
                row = [f"rho={rho} boot={boot}: size {100*bm.size(n, D(model, 'null3', **kw), 0.05)['rate']:.1f}"]
                for st in (1.0, 0.5):
                    dg = D(model, "break", 240, **({} if st == 1.0 else {"strength": st}), **kw)
                    b = bm.run_cell(m, dg, S, auto_jobs(jobs))
                    l = bm.location(b, dg)
                    row.append(f"s={st} power {100*bm.power(b, dg, 0.05)['rate']:.0f} in24 {100*l['first']['within_24']:.0f} lag {l['first']['median_lag']}")
                print(" | ".join(row), flush=True)
    elif group == "gatedtrim":
        # dating settings of the selected method: trimming and grid step (the gate is unchanged, so only dating moves)
        for trim in (0.15, 0.2):
            evaluate(f"gated split a=0.05 trim={trim}", M.GatedSplit(3, 0.05, trim=trim), model, S, jobs)
        evaluate("gated split a=0.05 step3", M.GatedSplit(3, 0.05, step=3), model, S, jobs)
