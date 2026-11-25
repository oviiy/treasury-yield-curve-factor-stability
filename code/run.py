"""Public reproduction commands: benchmark, synthetic data, and figures.

Figures of published Monte Carlo results read the frozen tables. Running a
small benchmark does not overwrite those results or imply a new estimate.
"""
import argparse
import csv
import re
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import simulation as sim

ROOT = Path(__file__).resolve().parents[1]
NAVY, TEAL, CORAL, GOLD, GREY = "#172554", "#087f78", "#c26040", "#bd8a2a", "#586579"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9aa5b5", "axes.titleweight": "bold",
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def table(letter):
    """Read a named Markdown table from the frozen week-nine results."""
    text = (ROOT / "results/final_tables.md").read_text(encoding="utf-8")
    section = re.search(rf"(?m)^## 表 {letter}(?:\s|（).*?(?=^## |\Z)", text, re.S | re.M)
    if section is None:
        raise ValueError(f"Table {letter} absent")
    lines = [line for line in section.group().splitlines() if line.startswith("|")]
    return [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines[2:]]


def save(fig, name, caption, bottom=.045):
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.text(.02, .012, caption, fontsize=8, color=GREY)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(out / name, dpi=180)
    plt.close(fig)
    print(out / name)


def figure_yields():
    df = pd.read_csv(ROOT / "data/treasury_cmt_h15_monthly_1962_2026.csv", parse_dates=["date"])
    df = df[df.date <= "2026-08-31"]
    cols = ["m3", "m6", "m12", "m24", "m36", "m60", "m84", "m120"]
    years = [1982, 1993, 2007, 2023]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4))
    for year, color in zip(years, [CORAL, GREY, TEAL, NAVY]):
        sub = df[df.date.dt.year == year][cols].mean()
        ax.plot([.25, .5, 1, 2, 3, 5, 7, 10], sub, marker="o", lw=2, color=color, label=str(year))
    ax.set(xlabel="Maturity (years)", ylabel="Yield (%)", title="Treasury curve across four episodes")
    ax.legend(ncol=2)
    for col, color, lab in [("m3", CORAL, "3 months"), ("m120", NAVY, "10 years")]:
        bx.plot(df.date, df[col], lw=1.15, color=color, label=lab)
    bx.set(xlabel="Date", ylabel="Yield (%)", title="H.15 monthly curve: short and long maturities")
    bx.legend()
    save(fig, "01_treasury_yields.png", "Source: Federal Reserve H.15 monthly CMT panel; available observations only.")


def figure_design():
    model, _ = sim.load_model("jkv")
    maturity = np.array([3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 48, 60, 72, 84, 96, 108, 120]) / 12
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4))
    for i, (name, color) in enumerate(zip(["level", "slope", "curvature"], [NAVY, TEAL, GOLD])):
        ax.plot(maturity, model[3][0][:, i], label=name, color=color, lw=2.1)
    ax.plot(maturity, model[4][0][:, 3], label="added fourth", color=CORAL, lw=2.3, ls="--")
    ax.set(xlabel="Maturity (years)", ylabel="Loading", title="Calibrated factor shapes")
    ax.legend()
    seed = sim.seeds("tune", 1)[0]
    _, null_signal = sim.panel(model, 480, "null3", 240, seed, return_signal=True)
    for tau, color in zip([160, 240, 320], [TEAL, NAVY, CORAL]):
        _, signal = sim.panel(model, 480, "break", tau, seed, return_signal=True)
        bx.plot(np.arange(480), 100 * (signal - null_signal)[:, 7], lw=1.2, label=f"month {tau}", color=color)
        bx.axvline(tau, color=color, ls=":", lw=1)
    bx.set(xlabel="Month", ylabel="Added signal (basis points)", title="Fourth factor enters at a known month")
    bx.legend()
    save(fig, "02_simulation_design.png", "JKV-calibrated synthetic signal; three planted dates. The real JKV panel is not redistributed.")


def figure_fullsample():
    rows = table("G")
    full = {int(re.search(r"@(\d+)%", r[0]).group(1)): r for r in rows if r[0].startswith("Full-sample")}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4))
    levels = [1, 5, 10]
    size = [float(full[a][1]) for a in levels]
    ax.bar(np.arange(3) - .15, size, .3, color=NAVY, label="Observed size")
    ax.bar(np.arange(3) + .15, levels, .3, color="#d5dee9", label="Nominal level")
    ax.set(xticks=np.arange(3), xticklabels=[f"{x}%" for x in levels], ylabel="Rejection rate (%)", title="Size: no planted break")
    ax.legend()
    x = np.arange(3)
    bx.plot(x, [float(v) for v in full[5][3:6]], color=NAVY, marker="o", lw=2.5, label="Full strength")
    bx.plot(x, [float(v) for v in full[5][6:9]], color=CORAL, marker="o", lw=2.5, label="Half strength")
    bx.set(xticks=x, xticklabels=["1/3", "1/2", "2/3"], ylim=(65, 105), xlabel="Planted break position in sample", ylabel="Rejection rate (%)", title="Power: three to four factors, 5% test")
    bx.legend()
    save(fig, "03_fullsample_size_power.png", "Source: frozen Table G, 500 final-seed panels per cell; full-sample rejection, no date estimate.")


def figure_bandwidth():
    rows = table("C")
    x = np.array([float(r[0]) for r in rows])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4))
    for col, level, color in [(1, 1, GOLD), (2, 5, NAVY), (3, 10, TEAL)]:
        ax.plot(x, [float(r[col]) for r in rows], marker="o", lw=2, color=color, label=f"Nominal {level}%")
        ax.axhline(level, color=color, lw=.9, ls=":")
    ax.set(xlabel="Bandwidth multiplier", ylabel="Observed size (%)", title="Size across bandwidth choices")
    ax.legend()
    bx.plot(x, [float(r[5]) for r in rows], marker="o", lw=2, color=NAVY, label="Half-strength power")
    bx.plot(x, [float(r[4]) for r in rows], marker="o", lw=2, color=CORAL, label="Full-strength power")
    bx.set(xlabel="Bandwidth multiplier", ylabel="Power at 5% (%)", ylim=(75, 105), title="Power at middle break")
    bx.legend()
    for a in (ax, bx): a.axvline(1, color=GREY, ls="--", lw=1)
    save(fig, "04_bandwidth.png", "Source: frozen Table C, 300 tune-seed panels; dashed vertical = paper rule.")


def figure_ar():
    rows = table("H")
    groups = [("Paper bootstrap", "论文 bootstrap", NAVY),
              ("AR(1), global residual", "全局残差", CORAL),
              ("AR(1), local residual", "局部残差", TEAL)]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4))
    for label, match, color in groups:
        r = [z for z in rows if match in z[0] and "Gated-split" in z[0]]
        x = [float(z[1]) for z in r]
        ax.plot(x, [float(z[3]) for z in r], marker="o", lw=2.2, label=label, color=color)
        bx.plot(x, [float(z[7]) for z in r], marker="o", lw=2.2, label=label, color=color)
    ax.axhline(5, color=GREY, ls="--", lw=1)
    ax.set(title="Size under no break", ylabel="Rejection rate (%)", ylim=(0, 105))
    bx.set(title="Power with a half-strength fourth factor", ylabel="Detection rate (%)", ylim=(0, 105))
    for a in (ax, bx): a.set_xlabel("Simulated error AR(1) coefficient")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(.5, .055), fontsize=9)
    save(fig, "05_serial_dependence.png", "Source: frozen Table H; 200 tune-seed panels. Global/local are exploratory bootstrap variants.", bottom=.16)


def figure_dating():
    g = table("G")
    split = next(r for r in g if r[0] == "Gated-split @5%")
    full = next(r for r in g if r[0] == "Full-sample test @5%")
    b = table("B（强度 1）")
    scan = [r for r in b if r[0] == "Scan month-by-month, full h @1%"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4))
    x = np.arange(3)
    ax.plot(x, [float(z) for z in split[9:12]], color=TEAL, marker="o", lw=2.2, label="Test, then split")
    ax.plot(x, [float(r[9]) for r in scan], color=CORAL, marker="o", lw=2.2, label="Sequential scan")
    ax.set(xticks=x, xticklabels=["1/3", "1/2", "2/3"], xlabel="Planted break position", ylabel="Within ±24 months (%)", ylim=(0, 100), title="Dating accuracy, full-strength change")
    ax.legend()
    bx.bar(x - .2, [float(z) for z in full[3:6]], .4, color=NAVY, label="Full-sample rejection")
    bx.bar(x + .2, [float(z) for z in split[3:6]], .4, color=TEAL, label="Gated split detection")
    bx.set(xticks=x, xticklabels=["1/3", "1/2", "2/3"], xlabel="Planted break position", ylabel="Rate (%)", ylim=(0, 110), title="Detection is distinct from dating")
    bx.legend()
    save(fig, "06_dating.png", "Sources: frozen Tables B and G; different simulation counts, so compare patterns, not paired outcomes.")


# ---------------------------------------------------------------- serial-correlation workstream (figures 07-11)
SERIAL = ROOT / "results/serial"


def serial_csv(name):
    return pd.read_csv(SERIAL / name)


def figure_persistence():
    grid = serial_csv("bandwidth_real_grid.csv")
    test = serial_csv("bandwidth_original_test_summary.csv")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4))
    for panel, label, color in [("JKV_full_17", "JKV, 1970–2009", NAVY), ("CMT_full_8", "Treasury CMT, 1981–2026", CORAL)]:
        g = grid[(grid.panel == panel) & (grid["rank"] == 3) & (grid.scale > 0)].sort_values("scale")
        fixed = grid[(grid.panel == panel) & (grid["rank"] == 3) & (grid.scale == 0)].median_rho1.iloc[0]
        ax.plot(g.scale, g.median_rho1, marker="o", lw=2, color=color, label=f"{label}: local fit")
        ax.axhline(fixed, color=color, ls=":", lw=1.2, label=f"{label}: fixed loadings")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set(xscale="log", xticks=[.25, .5, 1, 2, 3], xticklabels=["0.25", "0.5", "1", "2", "3"], ylim=(0, .8),
           xlabel="Bandwidth multiplier c (h = c × rule)", ylabel="Median lag-1 autocorrelation",
           title="Real data: narrower h whitens residuals only partly")
    ax.legend(fontsize=8)
    for ds, color in [("JKV17", NAVY), ("Treasury8", CORAL)]:
        for sc, ls, lab in [("null3_ar03", "-", "AR(0.3) noise"), ("null3_iid", "--", "independent noise")]:
            t = test[(test.dataset == ds) & (test.scenario == sc)].sort_values("scale")
            bx.plot(t.scale, 100 * t.reject_rate, marker="o", lw=2, ls=ls, color=color, label=f"{ds}, {lab}")
    bx.axhline(5, color=GREY, ls="--", lw=1)
    bx.set(xscale="log", ylim=(0, 105), xlabel="Bandwidth multiplier c",
           ylabel="Rejection rate with no change (%)", title="Original test: narrower h raises false rejections")
    bx.set_xticks([.5, 1, 2], ["0.5", "1", "2"]); bx.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    bx.legend(fontsize=8)
    save(fig, "07_bandwidth_persistence.png", "Sources: results/serial/bandwidth_real_grid.csv (rank 3) and bandwidth_original_test_summary.csv (200 panels, T = 240, 5% level).")


def figure_suwang_dgp():
    d = serial_csv("suwang_dgp_replication.csv")
    cells = [("DGP1_iid", "IID errors"), ("DGP1_ar03", "AR(0.3) errors"), ("DGP1_ar06", "AR(0.6) errors")]
    methods = [("J_paper", "Su–Wang J, paper bootstrap", NAVY), ("SM1", "Authors' MATLAB SM1", GREY),
               ("GLS_J_wild", "AR(1)-filtered GLS-J", TEAL)]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    x = np.arange(len(cells))
    for j, (m, lab, color) in enumerate(methods):
        v = [float(d[(d.cell == c) & (d.method == m)].reject_pct.iloc[0]) for c, _ in cells]
        ax.bar(x + (j - 1) * .26, v, .26, color=color, label=lab)
        for xi, vi in zip(x + (j - 1) * .26, v):
            ax.text(xi, vi + 1.5, f"{vi:.1f}" if vi < 10 else f"{vi:.0f}", ha="center", fontsize=8, color=color)
    ax.axhline(5, color=CORAL, ls="--", lw=1)
    ax.set(xticks=x, xticklabels=[l for _, l in cells], ylim=(0, 112), ylabel="Rejection rate with constant loadings (%)",
           title="Su–Wang's own DGP 1 (N = T = 100, R = 2): size fails once errors persist")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "08_suwang_dgp_serial.png", "Source: results/serial/suwang_dgp_replication.csv; 200 panels per cell, B = 199, 5% level.", bottom=.06)


NULL_LABELS = {
    "null_iid": "IID", "null_ar03": "AR(0.3)", "null_ar06": "AR(0.6)", "null_ar09": "AR(0.9)",
    "null_hetero": "AR 0.2–0.7 by maturity", "null_ar2": "AR(2)", "null_arma11": "ARMA(1,1)",
    "null_meas": "Persistent + measurement", "null_cross_ar06": "Cross-maturity corr.",
    "null_crosschg_ar06": "Cross corr. switches on", "null_vol15_ar03": "Volatility ×1.5, all maturities",
    "null_r4_iid": "Rank 4, IID (r = 4)", "null_r4_ar06": "Rank 4, AR(0.6) (r = 4)", "null_r4_hetero": "Rank 4, hetero AR (r = 4)",
    "null_volslope_ar03": "Volatility ×1→×2 by maturity, AR(0.3)", "null_volslope_ar06": "Volatility ×1→×2 by maturity, AR(0.6)"}


def new_method_cells():
    s = serial_csv("simulation_all_runs_summary.csv")
    s = s[s.method == "gls_aic_h2"].drop_duplicates(["dataset", "scenario"], keep="last")
    return {(r.dataset, r.scenario): r.reject_pct for r in s.itertuples()}


def figure_new_size():
    cell = new_method_cells()
    keys = list(NULL_LABELS)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    y = np.arange(len(keys))[::-1]
    ax.axvspan(2.3, 7.7, color="#e8eef5", lw=0, label="Monte Carlo 95% band around 5% (250 panels)")
    ax.axvline(5, color=GREY, ls="--", lw=1)
    for ds, color, off in [("JKV17", NAVY, .14), ("Treasury8", CORAL, -.14)]:
        ax.scatter([cell[(ds, k)] for k in keys], y + off, color=color, s=34, zorder=3, label=ds)
    ax.set(yticks=y, yticklabels=[NULL_LABELS[k] for k in keys], xlim=(0, 17), xlabel="Rejection rate at 5% with constant loadings (%)",
           title="Sieve-AR GLS-J size across error designs")
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "09_sieve_gls_size.png", "Source: results/serial/simulation_all_runs_summary.csv (method gls_aic_h2), 250 panels, T = 240, B = 199. The two bottom rows are the open limit.", bottom=.05)


def figure_new_power():
    cell = new_method_cells()
    boot = serial_csv("ar_bootstrap_study_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), gridspec_kw={"width_ratios": [1, 1.15, 1.15]})
    ax = axes[0]
    strengths = [.25, .5, .75, 1.0]
    keys = ["alt_load_ar06_s025", "alt_load_ar06", "alt_load_ar06_s075", "alt_load_ar06_s1"]
    for ds, color in [("JKV17", NAVY), ("Treasury8", CORAL)]:
        ax.plot(strengths, [cell[(ds, k)] for k in keys], marker="o", lw=2.2, color=color, label=ds)
    ax.axhline(5, color=GREY, ls="--", lw=1)
    ax.set(xticks=strengths, xlabel="Loading-shift strength", ylabel="Rejection rate at 5% (%)",
           ylim=(0, 100), title="Sieve-AR GLS-J power, AR(0.6) noise")
    ax.legend(fontsize=8)
    noises = [("Independent", "null_iid", "alt_loading_iid", "null_iid", "alt_load_iid"),
              ("AR(0.3)", "null_ar03", "alt_loading_ar03", "null_ar03", "alt_load_ar03"),
              ("AR(0.6)", "null_ar06", "alt_loading_ar06", "null_ar06", "alt_load_ar06")]
    methods = [("Original test", "#b8c2d0"), ("+ AR(1) bootstrap", GOLD), ("Sieve-AR GLS-J", TEAL)]
    for bx, ds in zip(axes[1:], ("JKV17", "Treasury8")):
        g = lambda sc, m: 100 * float(boot[(boot.dataset == ds) & (boot.scenario == sc) & (boot.method == m)].rejection_rate.iloc[0])
        x = np.arange(3)
        for j, (lab, color) in enumerate(methods):
            sizes, powers = [], []
            for _, n0, a0, n1, a1 in noises:
                if j == 0: sz, pw = g(n0, "paper_iid"), g(a0, "paper_iid")
                elif j == 1: sz, pw = g(n0, "ar1_global"), g(a0, "ar1_global")
                else: sz, pw = cell[(ds, n1)], cell[(ds, a1)]
                sizes.append(sz); powers.append(pw)
            xs = x + (j - 1) * .27
            bars = bx.bar(xs, powers, .27, color=color, label=lab)
            for xi, pw, sz, bar in zip(xs, powers, sizes, bars):
                bad = sz > 10
                if bad: bar.set_hatch("//"); bar.set_edgecolor(CORAL)
                bx.text(xi, pw + 2, f"{sz:.1f}" if sz < 10 else f"{sz:.0f}", ha="center", fontsize=7.5,
                        color=CORAL if bad else GREY, fontweight="bold" if bad else "normal")
        bx.set(xticks=x, xticklabels=[n[0] for n in noises], ylim=(0, 112), ylabel="Power, half-strength loading shift (%)",
               title=f"{ds}: power (size % above bars)")
        bx.legend(fontsize=7.5, loc="upper left")
    save(fig, "10_sieve_gls_power.png", "Hatched coral bars: size above 10%, so that power is mostly false alarms. Sources: simulation_all_runs_summary.csv (250 panels), ar_bootstrap_study_summary.csv (200 panels).", bottom=.06)


def figure_empirical():
    froz = serial_csv("empirical_frozen_pmax6.csv")
    pub = serial_csv("empirical_treasury8_public.csv")
    pub = pub[pub.pmax == 6]
    samples = [("JKV17 1970-1989", froz), ("JKV17 1990-2009", froz), ("JKV17 1970-2009", froz),
               ("Treasury8 1990-2009", pub), ("Treasury8 1990-2026", pub), ("Treasury8 2010-2026", pub)]
    fig, ax = plt.subplots(figsize=(9.5, 4))
    x = np.arange(len(samples))
    for j, (r, color) in enumerate([(3, NAVY), (4, TEAL)]):
        v = [float(d[(d["sample"] == s) & (d["rank"] == r)].p_value.iloc[0]) for s, d in samples]
        ax.bar(x + (j - .5) * .36, v, .36, color=color, label=f"r = {r}")
        for xi, vi in zip(x + (j - .5) * .36, v):
            ax.text(xi, vi + .006, f"{vi:.3f}", ha="center", fontsize=7.5, color=color)
    ax.axhline(.05, color=CORAL, ls="--", lw=1.1, label="5% level")
    ax.set(xticks=x, xticklabels=[s.replace(" ", "\n", 1) for s, _ in samples], ylim=(0, .4), ylabel="Bootstrap p-value (B = 999)",
           title="Exploratory sieve-AR GLS-J p-values (not pre-registered)")
    ax.legend(fontsize=8)
    save(fig, "11_empirical_pvalues.png", "JKV: frozen run on the licensed panel (not redistributed). Treasury: rerun on the committed H.15 file through August 2026. 2010–2026 sits in the open-limit case.", bottom=.07)


def benchmark(args):
    model, _ = sim.load_model(args.dataset)
    common = dict(generator="w9", T=args.T, tau=args.tau, dataset=args.dataset, model=model, resid_rho=args.rho)
    null = dict(common, setup="null3")
    change = dict(common, setup="break", strength=args.strength)
    method = sim.FullSample(d=3, B=args.bootstrap)
    ss = sim.seeds("final", args.reps)
    size = sim.size(sim.run_cell(method, null, ss, jobs=args.jobs), null, args.alpha)
    power = sim.power(sim.run_cell(method, change, ss, jobs=args.jobs), change, args.alpha)
    print(f"dataset={args.dataset} T={args.T} N={model[3][0].shape[0]} rho={args.rho} tau={args.tau} strength={args.strength}")
    print(f"size={size['rate']:.1%} (MC s.e. {size['se']:.1%}); power={power['rate']:.1%} (MC s.e. {power['se']:.1%})")


def synthetic(args):
    model, _ = sim.load_model(args.dataset)
    y = sim.panel(model, args.T, args.setup, args.tau, args.seed, args.strength, args.rho)
    if args.dataset == "jkv":
        cols = [f"m{k}" for k in [3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 48, 60, 72, 84, 96, 108, 120]]
    else:
        cols = ["m3", "m6", "m12", "m24", "m36", "m60", "m84", "m120"]
    path = ROOT / "data/synthetic_jkv_example.csv" if args.dataset == "jkv" else ROOT / "data/synthetic_cmt_example.csv"
    pd.DataFrame(y, columns=cols).assign(month=np.arange(1, len(y) + 1)).to_csv(path, index=False, float_format="%.8f")
    print(path)


def serial_benchmark(args):
    import serial_robust as sr
    rows = sr.run_serial_cells(args.datasets.split(","), args.scenarios.split(","), args.reps, args.bootstrap, args.jobs)
    for r in rows:
        print(f"{r['dataset']:9s} {r['scenario']:22s} reject={r['reject_pct']:5.1f}% (MC s.e. {r['mc_se_pts']:.1f}) n={r['reps']}")


def serial_empirical(args):
    import serial_robust as sr
    for name, d in sr.treasury8_samples().items():
        for r in (3, 4):
            o = sr.empirical_test(d[sr.CMT8].to_numpy(float), r, boot=args.bootstrap, pmax=args.pmax)
            print(f"{name} T={len(d)} r={r} AR order={o['p_ar']} stat={o['stat']:.5f} p={o['p']:.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("benchmark", help="Run new Monte Carlo size and power cells")
    d = sub.add_parser("synthetic-data", help="Write a clearly labeled synthetic example panel")
    sub.add_parser("figures", help="Rebuild figures from committed data and frozen tables")
    sb = sub.add_parser("serial-benchmark", help="Size/power of the sieve-AR GLS-J test on serial-error DGPs")
    sb.add_argument("--datasets", default="JKV17,Treasury8")
    sb.add_argument("--scenarios", default="null_iid,null_ar06,alt_load_ar06")
    sb.add_argument("--reps", type=int, default=10)
    sb.add_argument("--bootstrap", type=int, default=199)
    sb.add_argument("--jobs", type=int, default=1)
    se = sub.add_parser("serial-empirical", help="Sieve-AR GLS-J on the committed Treasury CMT panel")
    se.add_argument("--bootstrap", type=int, default=999)
    se.add_argument("--pmax", type=int, default=6)
    for q in (b, d):
        q.add_argument("--dataset", choices=["jkv", "treasury_cmt"], default="jkv")
        q.add_argument("--T", type=int, default=480)
        q.add_argument("--tau", type=int, default=240)
        q.add_argument("--rho", type=float, default=0)
        q.add_argument("--strength", type=float, default=1)
    b.add_argument("--reps", type=int, default=10)
    b.add_argument("--bootstrap", type=int, default=49)
    b.add_argument("--jobs", type=int, default=1)
    b.add_argument("--alpha", type=float, default=.05)
    d.add_argument("--setup", choices=["null3", "null4", "break"], default="break")
    d.add_argument("--seed", type=int, default=100000)
    args = p.parse_args()
    if args.command == "figures":
        for fn in [figure_yields, figure_design, figure_fullsample, figure_bandwidth, figure_ar, figure_dating,
                   figure_persistence, figure_suwang_dgp, figure_new_size, figure_new_power, figure_empirical]: fn()
    elif args.command == "serial-benchmark":
        if args.reps < 1 or args.bootstrap < 1 or args.jobs < 1:
            p.error("Require positive reps/bootstrap/jobs")
        serial_benchmark(args)
    elif args.command == "serial-empirical":
        serial_empirical(args)
    else:
        if args.T < 40 or not 1 <= args.tau < args.T or not 0 <= args.rho < 1 or args.strength < 0:
            p.error("Require T >= 40, 1 <= tau < T, 0 <= rho < 1, and strength >= 0")
        if args.command == "benchmark":
            if args.reps < 1 or args.bootstrap < 1 or args.jobs < 1 or not 0 < args.alpha < 1:
                p.error("Require positive reps/bootstrap/jobs and 0 < alpha < 1")
            benchmark(args)
        else: synthetic(args)


if __name__ == "__main__": main()
