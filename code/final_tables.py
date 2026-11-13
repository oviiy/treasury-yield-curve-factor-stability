"""All week-9 tables from the cached cells (no computation): writes results/final_tables.md.

Methods compared on the same simulated JKV panels (T = 480, d = 3): FullSample (Su-Wang full-sample test), Scan (the wk8
month-by-month scan, two bandwidth rules), Scan2 (step-6 scan), gated scans (derived: scan breaks only where the
FullSample p < gate; identical to running the gate, since the gate uses the same data and bootstrap seed), Rolling,
SupSplit (own bootstrap), GatedSplit (the selected method). Missing cells print as "—".
"""
import hashlib
import json

import benchmark as bm
import methods as M

LV = (0.01, 0.05, 0.10)
TAUS = (160, 240, 320)
OUT = []


def D(model, setup, tau=240, dataset="jkv", **k):
    return {"generator": "w9", "setup": setup, "T": 480, "tau": tau, "model": model, "dataset": dataset, **k}


def get(method, dgp, S):
    spec = {"method": method.config(), "dgp": {k: v for k, v in dgp.items() if k != "model"}, "v": bm.VERSION}
    key = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    path = bm.W9 / f"results/cells/{key}.jsonl"
    if not path.exists():
        return None
    got = {r["seed"]: r for r in map(json.loads, path.read_text().splitlines())}
    out = [got[s] for s in S if s in got]
    return out if len(out) == len(S) else None


def gate(scan_out, full_out, g):
    """Gated scan from cached cells: keep the scan's breaks only where the full-sample p < g."""
    if scan_out is None or full_out is None:
        return None
    fp = {r["seed"]: r["p"] for r in full_out}
    return [{**r, "breaks": r["breaks"] if fp[r["seed"]] < g else [], "edge": r["edge"] and fp[r["seed"]] < g} for r in scan_out]


def pct(m, key="rate"):
    return "—" if m is None else f"{100 * m[key]:.1f}"


def strength_kw(st):
    return {} if st == 1.0 else {"strength": st}


def line(*cells):
    OUT.append("| " + " | ".join(str(c) for c in cells) + " |")


def main():
    model, _ = bm.load_model("jkv")
    T100, T200, T500 = bm.seeds("tune", 100), bm.seeds("tune", 200), bm.seeds("tune", 500)
    F500 = bm.seeds("final", 500)

    # methods: (label, kind, getter(dgp, S) -> outcomes, alpha used for metrics, panels)
    full = bm.FullSample(3)
    rows = []
    for a in LV:
        rows.append((f"Full-sample test @{a:.0%}", "exact", lambda dg, S: get(full, dg, S), a, T500, T200))
    for a in LV:
        rows.append((f"Gated-split @{a:.0%} (selected)", "m", lambda dg, S, a=a: get(M.GatedSplit(3, a), dg, S), a, T200, T200))
    for a in LV:
        rows.append((f"SupSplit own bootstrap @{a:.0%}", "m", lambda dg, S, a=a: get(M.SupSplit(3, a), dg, S), a, T200, T200))
    for a in LV:
        rows.append((f"SupSplit diag bootstrap @{a:.0%}", "m", lambda dg, S, a=a: get(M.SupSplit(3, a, boot="diag"), dg, S), a, T200, T200))
    for bw in ("silverman", "full"):
        rows.append((f"Scan month-by-month, {bw} h @1%", "m", lambda dg, S, bw=bw: get(bm.Scan(3, 0.01, bw), dg, S), 0.01, T100, T100))
    rows.append(("Scan step 6, full h @1%", "m", lambda dg, S: get(M.Scan2(3, 0.01, step=6), dg, S), 0.01, T100, T100))
    for g in (0.05, 0.01):
        rows.append((f"Gated scan (full test {g:.0%}, then month-by-month full-h scan @1%)", "m",
                     lambda dg, S, g=g: gate(get(bm.Scan(3, 0.01, "full"), dg, S), get(full, dg, S), g), 0.01, T100, T100))
    for w in (96, 120, 180):
        rows.append((f"Rolling window {w} m @1%", "m", lambda dg, S, w=w: get(M.Rolling(3, 0.01, w), dg, S), 0.01, T100, T100))

    OUT.append("# 第九周最终表（JKV 校准，T = 480，d = 3；百分比）\n")
    OUT.append("所有数字由 `scripts/final_tables.py` 从缓存的仿真结果直接读出。tune 种子 = 探索用，final 种子 = 只在定稿时用一次。\n")
    OUT.append("## 表 A　size（无断点数据 null3：报出断点或拒绝的比例）\n")
    line("方法", "n", "size %", "MC s.e.", "无断点数据上的平均断点数")
    line(*["---"] * 5)
    for label, kind, fn, a, Sn, Sb in rows:
        o = fn(D(model, "null3"), Sn)
        if o is None:
            line(label, "—", "—", "—", "—")
            continue
        sz = bm.size(o, D(model, "null3"), a)
        line(label, sz["n"], f"{100 * sz['rate']:.1f}", f"{100 * sz['se']:.1f}", f"{sz['mean_breaks']:.2f}")

    for st in (1.0, 0.5):
        OUT.append(f"\n## 表 B（强度 {st:g}）　power 与定位（断点 3→4 因子在 τ；power = 在 τ−12 之后报出断点）\n")
        line("方法", "τ", "n", "power %", "τ−12 前误报 %", "漏报 %", "滞后中位数（月）", "MAD", "±12 %", "±24 %", "±36 %", "额外断点/面板")
        line(*["---"] * 12)
        for label, kind, fn, a, Sn, Sb in rows:
            for tau in TAUS:
                dg = D(model, "break", tau, **strength_kw(st))
                o = fn(dg, Sb if st != 1.0 or kind != "exact" else T500)
                if o is None:
                    continue  # not run (month-by-month and gated scans: strength 1 only)
                p = bm.power(o, dg, a)
                if kind == "exact":
                    line(label, tau, p["n"], f"{100 * p['rate']:.1f}", "–", "–", "–", "–", "–", "–", "–", "–")
                    continue
                l = bm.location(o, dg)
                f = l["first"]
                line(label, tau, p["n"], f"{100 * p['rate']:.1f}", f"{100 * p['false_alarm_pre']:.0f}", f"{100 * l['missed']:.0f}",
                     f["median_lag"], f["mad"], f"{100 * f['within_12']:.0f}", f"{100 * f['within_24']:.0f}",
                     f"{100 * f['within_36']:.0f}", f"{l['mean_extra_breaks']:.2f}")

    OUT.append("\n## 表 C　带宽：全样本检验，h = c × 规则（tune 种子，300 个面板；断点 τ = 240）\n")
    line("c", "size 1%", "size 5%", "size 10%", "power s=1 @5%", "power s=0.5 @1%", "power s=0.5 @5%", "power s=0.5 @10%")
    line(*["---"] * 8)
    S300 = bm.seeds("tune", 300)
    for c in (0.5, 0.75, 1.0, 1.5, 2.0):
        m = bm.FullSample(3, c=c)
        n = get(m, D(model, "null3"), S300)
        b1 = get(m, D(model, "break", 240), S300)
        b5 = get(m, D(model, "break", 240, strength=0.5), S300)
        if None in (n, b1, b5):
            line(c, *["—"] * 7)
            continue
        line(c, *[pct(bm.size(n, D(model, "null3"), a)) for a in LV], pct(bm.power(b1, D(model, "break", 240), 0.05)),
             *[pct(bm.power(b5, D(model, "break", 240, strength=0.5), a)) for a in LV])

    OUT.append("\n## 表 D　带宽：步长 6 扫描 @1%（tune 种子，100 个面板；τ = 240）\n")
    line("带宽规则", "size %", "power s=1", "滞后中位数 s=1", "±24 % s=1", "power s=0.5", "滞后中位数 s=0.5", "±24 % s=0.5")
    line(*["---"] * 8)
    for name, kw in (("full-sample T·h", {"bw_mode": "full"}), ("每窗口 Silverman", {"bw_mode": "silverman"}),
                     ("0.5× Silverman", {"bw_mode": "scale", "bw_value": 0.5}), ("2× Silverman", {"bw_mode": "scale", "bw_value": 2.0})):
        m = M.Scan2(3, 0.01, step=6, **kw)
        n = get(m, D(model, "null3"), T100)
        cells = []
        for st in (1.0, 0.5):
            dg = D(model, "break", 240, **strength_kw(st))
            o = get(m, dg, T100)
            if o is None:
                cells += ["—"] * 3
                continue
            l = bm.location(o, dg)
            cells += [pct(bm.power(o, dg, 0.01)), l["first"]["median_lag"], f"{100 * l['first']['within_24']:.0f}"]
        line(name, pct(bm.size(n, D(model, "null3"), 0.01)) if n else "—", *cells)

    OUT.append("\n## 表 E　步长 6 是否等于逐月扫描（同一批种子，全样本 T·h，1%）\n")
    line("数据", "n", "第一个断点完全相同 %", "相差 ≤ 6 个月 %", "是否报出断点一致 %")
    line(*["---"] * 5)
    for label, dg in [("null3", D(model, "null3"))] + [(f"break τ={t} s={st:g}", D(model, "break", t, **strength_kw(st)))
                                                         for st in (1.0, 0.5) for t in TAUS]:
        a, b = get(bm.Scan(3, 0.01, "full"), dg, T100), get(M.Scan2(3, 0.01, step=6), dg, T100)
        if a is None or b is None:
            continue  # month-by-month scan: strength 1 only
        f1 = [r["breaks"][0] if r["breaks"] else None for r in a]
        f2 = [r["breaks"][0] if r["breaks"] else None for r in b]
        same = sum(x == y for x, y in zip(f1, f2)) / len(a)
        near = sum((x is None and y is None) or (x is not None and y is not None and abs(x - y) <= 6) for x, y in zip(f1, f2)) / len(a)
        agree = sum((bool(r["breaks"]) or r["edge"]) == (bool(q["breaks"]) or q["edge"]) for r, q in zip(a, b)) / len(a)
        line(label, len(a), f"{100 * same:.0f}", f"{100 * near:.0f}", f"{100 * agree:.0f}")

    OUT.append("\n## 表 F　选定方法的定日期设定（Gated-split @5%，tune 200）\n")
    line("设定", "τ", "±24 % s=1", "滞后中位数 s=1", "±24 % s=0.5", "滞后中位数 s=0.5")
    line(*["---"] * 6)
    for name, m in (("trim 0.10, step 6（默认）", M.GatedSplit(3, 0.05)), ("trim 0.15", M.GatedSplit(3, 0.05, trim=0.15)),
                    ("trim 0.20", M.GatedSplit(3, 0.05, trim=0.2)), ("step 3", M.GatedSplit(3, 0.05, step=3)),
                    ("递归（多断点）", M.GatedSplit(3, 0.05, recursive=True))):
        for tau in TAUS:
            cells = []
            for st in (1.0, 0.5):
                dg = D(model, "break", tau, **strength_kw(st))
                o = get(m, dg, T200)
                if o is None:
                    cells += ["—", "—"]
                    continue
                l = bm.location(o, dg)
                cells += [f"{100 * l['first']['within_24']:.0f}", l["first"]["median_lag"]]
            line(name, tau, *cells)

    OUT.append("\n## 表 G　定稿：final 种子（从未用于选方法），500 个面板\n")
    line("方法", "size %", "s.e.", *[f"power s={st:g} τ={t}" for st in (1.0, 0.5) for t in TAUS], *[f"±24 s={st:g} τ={t}" for st in (1.0, 0.5) for t in TAUS])
    line(*["---"] * 15)
    for a in (0.05, 0.01):
        m = M.GatedSplit(3, a)
        n = get(m, D(model, "null3"), F500)
        cells, locs = [], []
        for st in (1.0, 0.5):
            for t in TAUS:
                dg = D(model, "break", t, **strength_kw(st))
                o = get(m, dg, F500)
                cells.append(pct(bm.power(o, dg, a)) if o else "—")
                locs.append(f"{100 * bm.location(o, dg)['first']['within_24']:.0f}" if o else "—")
        sz = bm.size(n, D(model, "null3"), a) if n else None
        line(f"Gated-split @{a:.0%}", pct(sz), pct(sz, "se"), *cells, *locs)
    n = get(full, D(model, "null3"), F500)
    for a in LV:
        cells = []
        for st in (1.0, 0.5):
            for t in TAUS:
                dg = D(model, "break", t, **strength_kw(st))
                o = get(full, dg, F500)
                cells.append(pct(bm.power(o, dg, a)) if o else "—")
        sz = bm.size(n, D(model, "null3"), a) if n else None
        line(f"Full-sample test @{a:.0%}", pct(sz), pct(sz, "se"), *cells, *["–"] * 6)

    OUT.append("\n## 表 H　残差有自相关（AR(1)，系数 ρ）时的 size 与 power（tune 种子；τ = 240）\n")
    OUT.append("论文的 bootstrap 生成时间上独立的噪声；“AR(1) bootstrap” 按每个期限的残差一阶自相关生成噪声（我加的，非论文做法）。初版从全局（固定载荷）拟合的残差估计自相关；有断点时这个残差里还含有漏掉的第四因子，自相关被高估，power 被机械地压低（独立审查发现），修正版改用局部拟合的残差、允许负值：power 回升，但 size 超标（局部拟合吸收了一部分噪声，自相关和噪声方差都被低估）。两个版本都不理想，怎么处理需要和老师讨论。\n")
    line("方法", "ρ", "n", "size %", "power s=1", "±24 % s=1", "滞后中位数 s=1", "power s=0.5", "±24 % s=0.5")
    line(*["---"] * 9)
    names = {"paper": "论文", "ar1": "AR(1)，全局残差（初版）", "ar1loc": "AR(1)，局部残差（审查后修正）"}
    for boot in ("paper", "ar1", "ar1loc"):
        m = M.GatedSplit(3, 0.05, boot=boot)
        for rho in (0.0, 0.3, 0.5, 0.7):
            kw = {} if rho == 0 else {"resid_rho": rho}
            n = get(m, D(model, "null3", **kw), T200)
            if n is None:
                continue
            cells = []
            for st in (1.0, 0.5):
                dg = D(model, "break", 240, **strength_kw(st), **kw)
                o = get(m, dg, T200)
                l = bm.location(o, dg)["first"]
                cells += [pct(bm.power(o, dg, 0.05)), f"{100 * l['within_24']:.0f}"] + ([l["median_lag"]] if st == 1.0 else [])
            line(f"Gated-split @5%, {names[boot]} bootstrap", rho, len(n),
                 pct(bm.size(n, D(model, "null3", **kw), 0.05)), *cells)
    note = ""
    for rho in (0.0, 0.5):
        kw = {} if rho == 0 else {"resid_rho": rho}
        n = get(M.Scan2(3, 0.01, step=6), D(model, "null3", **kw), T100)
        if n is not None:
            sz = bm.size(n, D(model, "null3", **kw), 0.01)
            line("Scan step 6, full h @1%（论文 bootstrap）", rho, len(n), pct(sz), *["–"] * 5)
            if rho:
                note = f"\n扫描在 ρ = {rho} 时，无断点数据上平均报出 {sz['mean_breaks']:.2f} 个断点。"

    OUT.append(note)
    OUT.append("\n## 表 I　压力测试：选定方法 Gated-split @5%（tune 种子，200 个面板）\n")
    line("设定", "size %", "强度", "τ", "power %", "滞后中位数", "MAD", "±24 %", "第一个断点在 τ−12 前 %")
    line(*["---"] * 9)
    tsy, _ = bm.load_model("treasury_cmt")
    m = M.GatedSplit(3, 0.05)
    for name, mdl, ds, kw, sts in (("JKV，强度加密", model, "jkv", {}, (0.3, 0.4, 0.5, 0.6, 0.75, 1.0)),
                                   ("噪声取四因子拟合的残差", model, "jkv", {"resid_from": 4}, (1.0, 0.5)),
                                   ("财政部 CMT 校准（N = 8）", tsy, "treasury_cmt", {}, (1.0, 0.5))):
        n = get(m, D(mdl, "null3", dataset=ds, **kw), T200)
        szs = pct(bm.size(n, D(mdl, "null3", dataset=ds, **kw), 0.05)) if n else "—"
        for st in sts:
            for tau in TAUS:
                dg = D(mdl, "break", tau, dataset=ds, **strength_kw(st), **kw)
                o = get(m, dg, T200)
                if o is None:
                    continue
                l = bm.location(o, dg)
                f = l["first"]
                line(name, szs, st, tau, pct(bm.power(o, dg, 0.05)), f["median_lag"], f["mad"], f"{100 * f['within_24']:.0f}",
                     f"{100 * l['early_false']:.0f}")

    (bm.W9 / "results/final_tables.md").write_text("\n".join(OUT) + "\n")
    print("\n".join(OUT))


if __name__ == "__main__":
    main()
