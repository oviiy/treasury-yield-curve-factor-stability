"""Public reproduction commands: benchmark, synthetic data, and figures.

Figures of published Monte Carlo results read the frozen tables. Running a
small benchmark does not overwrite those results or imply a new estimate.
"""
import argparse
import csv
import re
from pathlib import Path

import matplotlib
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("benchmark", help="Run new Monte Carlo size and power cells")
    d = sub.add_parser("synthetic-data", help="Write a clearly labeled synthetic example panel")
    sub.add_parser("figures", help="Rebuild figures from committed data and frozen tables")
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
        for fn in [figure_yields, figure_design, figure_fullsample, figure_bandwidth, figure_ar, figure_dating]: fn()
    else:
        if args.T < 40 or not 1 <= args.tau < args.T or not 0 <= args.rho < 1 or args.strength < 0:
            p.error("Require T >= 40, 1 <= tau < T, 0 <= rho < 1, and strength >= 0")
        if args.command == "benchmark":
            if args.reps < 1 or args.bootstrap < 1 or args.jobs < 1 or not 0 < args.alpha < 1:
                p.error("Require positive reps/bootstrap/jobs and 0 < alpha < 1")
            benchmark(args)
        else: synthetic(args)


if __name__ == "__main__": main()
