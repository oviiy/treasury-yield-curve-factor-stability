"""Multi-break sequential scan (advisor D3) on JKV 1970-2009 and Treasury 1990-2026."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from wk4_common import WK4, silverman_th, TREASURY_METHOD_CHANGE, load_jkv, load_treasury, load_treasury_cmt_long, run_test
from segmented_scan import segmented_scan


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


def to_dates(segments, dates):
    iso = lambda i: str(dates.iloc[i].date())
    return [{"start_date": iso(g["start"]),
             "break_date": None if g["break_month"] is None else iso(g["break_month"]),
             "rejected_at_min_window": g["rejected_at_min_window"],
             "retreated": g.get("retreated", False), "break_p": g.get("break_p"),
             "trajectory": [{**r, "end_date": iso(r["end"] - 1)} for r in g["trajectory"]]}
            for g in segments]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["jkv", "treasury", "treasury_cmt"], required=True)
    p.add_argument("--ranks", type=int, nargs="+", default=[3])
    p.add_argument("--min-window", type=int, default=60,
                    help="minimum window for the first segment (advisor, 2026-09-29: 60, was 100)")
    p.add_argument("--restart-min-window", type=int, default=None,
                    help="minimum window for segments after a break (advisor: can be much shorter, e.g. 12; "
                         "defaults to --min-window)")
    p.add_argument("--no-retreat", action="store_true",
                    help="disable the 2026-09-29 retreat fix and reproduce the old (buggy) EDGE behaviour")
    p.add_argument("--alpha", type=float, default=0.01,
                    help="significance level of each window test (add-one bootstrap p < alpha); default 1%%")
    p.add_argument("--bw-mode", choices=["silverman", "fixed_h", "fixed_th", "scale", "full"], default="silverman",
                    help="week 7: silverman = h recomputed from each window's length (default, the wk4-6 behaviour); "
                         "fixed_h = same h in every window; fixed_th = same T*h (months) in every window; scale = c x Silverman; full = the full-sample T*h in every window (week 8)")
    p.add_argument("--bw-value", type=float, default=None, help="h for fixed_h (e.g. 0.149) or T*h in months for fixed_th (e.g. 36)")
    p.add_argument("--run-k", type=int, default=1,
                    help="consecutive rejecting windows needed for a break (week 6); 1 = the original rule")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--quick", action="store_true", help="199 screen draws (min add-one p = 0.005), no confirm; smoke test only")
    a = p.parse_args()
    reps = {"screen_reps": 199, "confirm_reps": 0} if a.quick else {}
    reps["alpha"] = a.alpha
    reps.update(bw_mode=a.bw_mode, bw_value=a.bw_value)
    dates, y = {"jkv": load_jkv, "treasury": load_treasury, "treasury_cmt": load_treasury_cmt_long}[a.dataset]()
    if a.bw_mode == "full" and a.bw_value is None:
        a.bw_value = round(float(silverman_th(*y.shape)), 1)  # JKV 71.4, Treasury CMT 84.5
        reps.update(bw_value=a.bw_value)
    if a.bw_mode != "silverman" and a.bw_value is None:
        p.error("--bw-value is required with fixed_h / fixed_th / scale")
    (WK4 / "results").mkdir(exist_ok=True)
    restart_mw = a.restart_min_window if a.restart_min_window is not None else a.min_window
    suffix = "" if a.min_window == 100 and restart_mw == 100 else f"_mw{a.min_window}"
    if a.restart_min_window is not None:
        suffix += f"_restart{restart_mw}"
    if a.alpha != 0.01:
        suffix += f"_a{a.alpha:g}"
    if a.run_k != 1:
        suffix += f"_k{a.run_k}"
    if a.bw_mode != "silverman":
        suffix += f"_{a.bw_mode}{a.bw_value:g}"
    if a.quick:
        suffix += "_quick"
    by_rank = {}
    with ProcessPoolExecutor(a.jobs) as pool:
        for rank in a.ranks:
            segs = segmented_scan(len(y), WindowTester(y, rank, **reps), a.min_window,
                                   restart_min_window=restart_mw, retreat=not a.no_retreat, run_k=a.run_k,
                                   batch=a.jobs, mapper=pool.map)
            by_rank[rank] = to_dates(segs, dates)
            spec = {"dataset": a.dataset, "rank": rank, "T": len(y), "N": y.shape[1],
                    "sample": [str(dates.iloc[0].date()), str(dates.iloc[-1].date())],
                    "min_window": a.min_window, "restart_min_window": restart_mw, "retreat": not a.no_retreat,
                    "alpha": a.alpha, "run_k": a.run_k, "screen_reps": reps.get("screen_reps", 999),
                    "confirm_reps": reps.get("confirm_reps", 9999),
                    "standardization": "within each window", "bandwidth": "own h per window" if a.bw_mode == "silverman" else (
                        f"full-sample T*h = {a.bw_value} months in every window" if a.bw_mode == "full" else f"{a.bw_mode} = {a.bw_value}"),
                    "break_rule": f"first month whose inclusion rejects at {a.alpha:.2%}; next segment starts at that month; "
                                  "if the very first test already rejects, retreat to the true non-rejecting "
                                  "boundary, record that boundary as the break and restart there (advisor, 2026-09-29)"}
            out = WK4 / f"results/multibreak_{a.dataset}_d{rank}{suffix}.json"
            out.write_text(json.dumps({"spec": spec, "segments": by_rank[rank]}, indent=2))
            print(rank, [g["break_date"] for g in by_rank[rank]],
                  ["EDGE" if g["rejected_at_min_window"] else "" for g in by_rank[rank]], flush=True)


if __name__ == "__main__":
    main()
