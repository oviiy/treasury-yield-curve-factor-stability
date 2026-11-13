"""Figures for the week-9 deck (academic style, deck palette). Reads data and cached simulation cells only.
Usage: python3 scripts/make_wk9_figures.py  ->  wk9/figures/fig_*.png"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import benchmark as bm
import final_tables as FT
import methods as M
from wk4_common import load_jkv, load_treasury_cmt_long

NAVY, TEAL, TERRA, GREY, INK, CREAM = "#172554", "#0F766E", "#C65D3A", "#5E6878", "#152033", "#F6F3EC"
GOLD, SKY = "#B8860B", "#4A7FB5"
FIG = bm.W9 / "figures"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 11,
                     "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": NAVY, "axes.titlelocation": "left",
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 10,
                     "figure.facecolor": CREAM, "axes.facecolor": CREAM, "savefig.facecolor": CREAM})
model, _ = bm.load_model("jkv")
D, get = FT.D, FT.get
TAUS = (160, 240, 320)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"fig_{name}.png", dpi=220)
    plt.close(fig)


def curves():
    eras = (("1981", "1981-09", "1981-12", TERRA), ("1993", "1993-01", "1993-12", GREY), ("2007", "2007-01", "2007-12", TEAL),
            ("2023", "2023-01", "2023-12", NAVY))
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.3), sharey=True)
    for ax, (name, loader, mats) in zip(axes, (("JKV (zero-coupon, N = 17)", load_jkv, bm.sim.maturities("jkv")),
                                              ("Treasury CMT (par, N = 8)", load_treasury_cmt_long, bm.sim.maturities("treasury_cmt")))):
        d, y = loader()
        d = np.array(d.dt.strftime("%Y-%m"))
        for lab, a, b, c in eras:
            m = (d >= a) & (d <= b)
            if m.any():
                ax.plot(np.array(mats) / 12, y[m].mean(0), color=c, lw=2, marker="o", ms=4, label=lab)
        ax.set_title(name)
        ax.set_xlabel("maturity (years)")
        ax.legend(loc="center right", ncol=2, bbox_to_anchor=(1, 0.55), handlelength=1.2, columnspacing=0.8)
    axes[0].set_ylabel("average yield (%)")
    save(fig, "curves")


def dgp():
    fig, axes = plt.subplots(1, 2, figsize=(7.9, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})
    mats = np.array(bm.sim.maturities("jkv")) / 12
    lam4 = model[4][0]
    for j, (lab, c) in enumerate((("level", NAVY), ("slope", TEAL), ("curvature", GOLD), ("4th (Svensson)", TERRA))):
        axes[0].plot(mats, lam4[:, j], color=c, lw=2.8 if j == 3 else 2.2, ls="-" if j < 3 else "--", label=lab)
    axes[0].set_title("Loadings by maturity")
    axes[0].set_xlabel("maturity (years)")
    axes[0].legend(loc="center right")
    seed = bm.seeds("tune", 1)[0]
    _, s0 = bm.panel(model, 480, "null3", 240, seed, return_signal=True)
    for tau, c in zip(TAUS, (TEAL, NAVY, TERRA)):
        _, s1 = bm.panel(model, 480, "break", tau, seed, return_signal=True)
        axes[1].plot(np.arange(480), (s1 - s0)[:, 4] * 100, color=c, lw=1.2, label=f"τ = {tau}")
        axes[1].axvline(tau, color=c, lw=1, ls=":")
    axes[1].set_title("Added 4th factor, 2-year yield (bp)")
    axes[1].set_xlabel("month")
    axes[1].legend(loc="lower left", ncol=3, fontsize=12, handlelength=1.2, columnspacing=0.8)
    axes[0].legend(loc="center right", fontsize=12)
    save(fig, "dgp")


def kernel():
    fig, ax = plt.subplots(figsize=(5.9, 2.2))
    T, th = 480, bm.silverman_th(480, 17)
    t = np.arange(1, T + 1)
    for r, c, lab in ((240, NAVY, "weights at month 240"), (30, TEAL, "month 30 (boundary-corrected)")):
        u = (t - r) / th
        k = .75 * np.maximum(1 - u * u, 0)
        k = k / k.sum()
        ax.fill_between(t, k, color=c, alpha=.18)
        ax.plot(t, k, color=c, lw=2.4, label=lab)
    ax.annotate("", xy=(240 + th, 0.0005), xytext=(240, 0.0005), arrowprops=dict(arrowstyle="<->", color=INK))
    ax.text(240 + th / 2, 0.0012, f"T·h ≈ {th:.0f} months", ha="center", color=INK)
    ax.set_xlabel("month")
    ax.set_ylabel("weight")
    ax.set_yticks([])
    ax.legend(loc="upper right")
    save(fig, "kernel")


def fullsample():
    full = bm.FullSample(3)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(3)
    for k, (blk, c) in enumerate((("tune", SKY), ("final", NAVY))):
        o = get(full, D(model, "null3"), bm.seeds(blk, 500))
        r = [100 * bm.size(o, D(model, "null3"), a)["rate"] for a in FT.LV]
        se = [100 * bm.size(o, D(model, "null3"), a)["se"] for a in FT.LV]
        axes[0].bar(x + (k - .5) * .34, r, .32, color=c, label=f"{blk} seeds (500)")
        for xi, v in zip(x, r):
            axes[0].text(xi + (k - .5) * .34, v + 0.25, f"{v:.1f}", ha="center", va="bottom", fontsize=13, color=INK)
    for xi, a in zip(x, (1, 5, 10)):
        axes[0].hlines(a, xi - .4, xi + .4, color=TERRA, lw=2.5)
    axes[0].set_xticks(x, ["1%", "5%", "10%"])
    axes[0].set_ylim(0, 14)
    axes[0].set_ylabel("rejection rate (%)")
    axes[0].set_title("Size: no break (red = nominal level)")
    axes[0].legend(loc="upper left")
    for st, ls in ((1.0, "-"), (0.5, "--")):
        for a, c in ((0.01, TERRA), (0.05, TEAL), (0.10, NAVY)):
            S = bm.seeds("tune", 500 if st == 1 else 200)
            pw = [100 * bm.power(get(full, D(model, "break", t, **FT.strength_kw(st)), S), D(model, "break", t, **FT.strength_kw(st)), a)["rate"]
                  for t in TAUS]
            axes[1].plot(["1/3", "1/2", "2/3"], pw, color=c, ls=ls, lw=2.4, marker="o", ms=7,
                         label=f"{int(a * 100)}%" if st == 1 else None)
    axes[1].text(0, 60.5, "half strength (dashed)", fontsize=12, color=GREY, ha="left")
    axes[1].text(0, 101.5, "full strength (solid) ≈ 100%", fontsize=12, color=GREY)
    axes[1].set_ylim(55, 106)
    axes[1].set_ylabel("rejection rate (%)")
    axes[1].set_xlabel("break position τ / T")
    axes[1].set_title("Power: 3 to 4 factors at τ")
    axes[1].legend(loc="lower right", title="level", ncol=3)
    save(fig, "fullsample")


def bandwidth():
    S = bm.seeds("tune", 300)
    cs = (0.5, 0.75, 1.0, 1.5, 2.0)
    sz = {a: [] for a in FT.LV}
    pw = {a: [] for a in FT.LV}
    for c in cs:
        m = bm.FullSample(3, c=c)
        n, b = get(m, D(model, "null3"), S), get(m, D(model, "break", 240, strength=0.5), S)
        for a in FT.LV:
            sz[a].append(100 * bm.size(n, D(model, "null3"), a)["rate"])
            pw[a].append(100 * bm.power(b, D(model, "break", 240, strength=0.5), a)["rate"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for a, c in ((0.01, TERRA), (0.05, TEAL), (0.10, NAVY)):
        axes[0].plot(cs, sz[a], color=c, lw=2.4, marker="o", ms=7, label=f"{int(a * 100)}%")
        axes[0].axhline(a * 100, color=c, lw=1, ls=":")
        axes[1].plot(cs, pw[a], color=c, lw=2.4, marker="o", ms=7, label=f"{int(a * 100)}%")
    for ax, t in zip(axes, ("Size (dotted = nominal)", "Power, half-strength factor, τ = T/2")):
        ax.axvline(1, color=GREY, lw=1, ls="--")
        ax.set_xticks(cs, ["0.5", "0.75", "1\n(paper)", "1.5", "2"])
        ax.set_xlabel("bandwidth multiplier c")
        ax.set_title(t)
        ax.set_ylabel("rejection rate (%)")
    axes[0].legend(title="level")
    save(fig, "bandwidth")


def arnoise():
    names = {"paper": ("Su–Wang bootstrap (i.i.d. in time)", NAVY), "ar1": ("AR(1) bootstrap, global residual", TERRA),
             "ar1loc": ("AR(1) bootstrap, local residual", TEAL)}
    rhos = (0.0, 0.3, 0.5, 0.7)
    S = bm.seeds("tune", 200)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for boot, (lab, c) in names.items():
        m = M.GatedSplit(3, 0.05, boot=boot)
        sz, pw = [], []
        for r in rhos:
            kw = {} if r == 0 else {"resid_rho": r}
            sz.append(100 * bm.size(get(m, D(model, "null3", **kw), S), D(model, "null3", **kw), 0.05)["rate"])
            dg = D(model, "break", 240, strength=0.5, **kw)
            pw.append(100 * bm.power(get(m, dg, S), dg, 0.05)["rate"])
        axes[0].plot(rhos, sz, color=c, lw=2.6, marker="o", ms=8, label=lab)
        if boot != "paper":
            axes[1].plot(rhos, pw, color=c, lw=2.6, marker="o", ms=8, label=lab)
    axes[0].axhline(5, color=GREY, lw=1.2, ls="--")
    axes[0].text(0.7, 8, "nominal 5%", ha="right", color=GREY, fontsize=12)
    axes[0].axvspan(0.55, 0.7, color=GOLD, alpha=.12)
    axes[0].text(0.625, 45, "real JKV\nresiduals\nρ ≈ 0.63", ha="center", color=GOLD, fontsize=12)
    axes[0].set_title("Size of the test at 5%")
    axes[1].set_title("Power, half-strength factor")
    for ax in axes:
        ax.set_xticks(rhos)
        ax.set_xlabel("autocorrelation of the noise ρ")
        ax.set_ylabel("rate (%)")
        ax.set_ylim(0, 105)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=13)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(FIG / "fig_arnoise.png", dpi=220)
    plt.close(fig)


def scanlag():
    dg = D(model, "break", 240)
    lags = {"Break scan, 1%": ([r["breaks"][0] - 240 for r in get(bm.Scan(3, 0.01, "full"), dg, bm.seeds("tune", 100)) if r["breaks"]], GREY),
            "Test, then split, 5%": ([r["breaks"][0] - 240 for r in get(M.GatedSplit(3, 0.05), dg, bm.seeds("tune", 200)) if r["breaks"]], TEAL)}
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    bins = np.arange(-120, 132, 12)
    for (lab, (lag, c)), off in zip(lags.items(), (-2.6, 2.6)):
        h, _ = np.histogram(np.clip(lag, -120, 119), bins=bins)
        ax.bar(bins[:-1] + 6 + off, 100 * h / len(lag), 5.2, color=c, label=lab)
    ax.axvspan(-24, 24, color=GOLD, alpha=.10)
    ax.axvline(0, color=INK, lw=1)
    ax.text(-3, ax.get_ylim()[1] * .92, "true break", ha="right", color=INK)
    ax.set_xlabel("first dated break − τ, months (break at T/2; shaded ±24)")
    ax.set_ylabel("% of panels")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.75))
    save(fig, "scanlag")


def bootdecision():
    import pandas as pd
    R5 = bm.W9 / "data"
    head = pd.read_csv(R5 / "headline_tests.csv")
    z = np.load(R5 / "j_distribution_draws.npz")
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.9))
    for i, (key, panel) in enumerate((("jkv", "JKV 1970-2009"), ("tsy", "Treasury CMT (H.15) 1981-09-2026-08"))):
        for j, d in enumerate((3, 4, 5)):
            ax = axes[i, j]
            r = head[(head.panel == panel) & (head.d == d)].iloc[0]
            draws = z[f"{key}_d{d}"]
            hi = max(np.quantile(draws, .999), r.critical_99) * 1.25
            ax.hist(draws[draws < hi], bins=30, color=SKY, alpha=.75)
            ax.tick_params(labelsize=9)
            ax.axvline(r.critical_99, color=INK, ls="--", lw=1.2)
            if r.J < hi:
                ax.axvline(r.J, color=TERRA, lw=2.2)
            else:
                ax.annotate(f"J = {r.J:.0f}, off scale", xy=(.98, .8), xycoords="axes fraction", ha="right", color=TERRA, fontsize=10,
                            bbox=dict(fc=CREAM, ec="none", pad=1.5))
            ax.set_yticks([])
            ax.set_title(f"{'JKV' if key == 'jkv' else 'Treasury'} d = {d}: {'reject' if r.reject_1pct else 'not rejected'}",
                         fontsize=11, color=NAVY if r.reject_1pct else GREY)
            ax.spines["left"].set_visible(False)
    save(fig, "bootdecision")


def datingillus():
    """Schematic of the scan, the least-squares split (from one simulated panel) and the dating metrics."""
    import newmethods_poc as NP
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.5), gridspec_kw={"width_ratios": [1.15, 1, 1.1]})
    ax = axes[0]
    tau = 240
    for k, (end, rej) in enumerate(((60, 0), (120, 0), (180, 0), (240, 0), (264, 1))):
        ax.plot([0, end], [k, k], color=TERRA if rej else GREY, lw=7, solid_capstyle="butt")
        ax.text(end + 6, k, "reject" if rej else "no", va="center", fontsize=10, color=TERRA if rej else GREY)
    ax.plot([264, 324], [5.2, 5.2], color=TEAL, lw=7, solid_capstyle="butt")
    ax.text(330, 5.2, "restart", va="center", fontsize=10, color=TEAL)
    ax.axvline(tau, color=INK, lw=1.2, ls="--")
    ax.text(tau - 4, 6.1, "true break", ha="right", fontsize=10)
    ax.annotate("dated here:\nlate", xy=(264, 4), xytext=(330, 2.2), fontsize=10, color=TERRA, arrowprops=dict(arrowstyle="->", color=TERRA))
    ax.set_xlim(0, 420); ax.set_ylim(-0.7, 6.6); ax.set_yticks([]); ax.set_xlabel("month")
    ax.set_title("Scan: grow the window until the test rejects")
    ax = axes[1]
    seed = bm.seeds("tune", 3)[2]
    for setup, c, lab in (("break", NAVY, "break at T/2"), ("null3", GREY, "no break")):
        y = bm.panel(model, 480, setup, 240, seed)
        g = NP.curves(y)["split"]
        k = np.array(sorted(g)); v = np.array([g[i] for i in k]) * 100
        ax.plot(k, v, color=c, lw=1.8, label=lab)
        if setup == "break":
            kb = k[np.argmax(v)]
            ax.plot([kb], [v.max()], "o", color=TERRA, ms=7)
            ax.annotate(f"date = month {kb}", xy=(kb, v.max()), xytext=(300, v.max() * 0.95), fontsize=10, color=TERRA, arrowprops=dict(arrowstyle="->", color=TERRA))
    ax.axvline(240, color=INK, lw=1.2, ls="--")
    ax.set_xlabel("candidate split month k"); ax.set_ylabel("fit gain from splitting (%)")
    ax.set_title("Split: the best split month is the date")
    ax.legend(loc="upper left", fontsize=10)
    ax = axes[2]
    ax.axvspan(tau - 24, tau + 24, color=GOLD, alpha=.15)
    ax.axvline(tau, color=INK, lw=1.2, ls="--")
    ax.axvline(tau - 12, color=TERRA, lw=1.2, ls=":")
    for xv, yv, lab, c, tx in ((tau - 60, 2.2, "early:\nfalse alarm", TERRA, tau - 88), (tau + 10, 1.4, "within ±24: accurate", TEAL, tau + 14),
                               (tau + 55, 0.6, "late: lag = +55", NAVY, tau + 59)):
        ax.plot([xv], [yv], "o", color=c, ms=8)
        ax.text(tx, yv + 0.22, lab, fontsize=10, color=c)
    ax.annotate("", xy=(tau + 55, 0.35), xytext=(tau, 0.35), arrowprops=dict(arrowstyle="<->", color=NAVY))
    ax.text(tau - 14, 3.0, "τ − 12", ha="right", fontsize=10, color=TERRA)
    ax.text(tau + 2, 3.0, "τ", fontsize=10)
    ax.set_xlim(150, 360); ax.set_ylim(0, 3.4); ax.set_yticks([]); ax.set_xlabel("first dated break (month)")
    ax.set_title("Metrics: power counts dates after τ − 12")
    save(fig, "datingillus")


def newpoc():
    S = np.load(bm.W9 / "results/newmethods_poc_summary.npy", allow_pickle=True).item()
    labs = (("split", "Least-squares split"), ("mosum96", "Two-sided windows, 96 m"), ("mosum60", "Two-sided windows, 60 m"),
            ("eigratio96", "4th/5th eigenvalue ratio"), ("subsp96", "Loading-subspace shift"))
    fig, ax = plt.subplots(figsize=(13, 4.6))
    y = np.arange(len(labs))[::-1]
    for k, (st, c, lab) in enumerate(((1.0, NAVY, "i.i.d. noise, full strength"), (0.5, SKY, "i.i.d. noise, half strength"),
                                       ("ar", TERRA, "AR noise ρ = .5, full strength"))):
        v = [S[key][(st, 240)][2] if st != "ar" else S[key]["ar"][2] for key, _ in labs]
        ax.barh(y + (1 - k) * .27, v, .25, color=c, label=lab)
    ax.set_yticks(y, [l for _, l in labs])
    ax.set_xlabel("% of panels dated within ±24 months of τ = T/2")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right")
    save(fig, "newpoc")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for f in (curves, dgp, kernel, bootdecision, scanlag, datingillus):
        f()
        print("ok", f.__name__, flush=True)
