"""Build the two current decks from committed figures and result tables.

    python slides/build_slides.py

Writes slides/Research_Update_Full_Storyline.pptx (main deck) and
slides/Weekly_Quick_Take_2026-11-25.pptx (seven-page weekly update).
Numbers are typed from results/final_tables.md and results/serial/*.csv;
each slide's notes name its source file.
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
ASSET = Path(__file__).resolve().parent / "assets"

PAPER, CARD, NAVY, TEAL = "F6F3EC", "FDFCF8", "172554", "0F766E"
INK, MUTED, MIST, PALE, RED = "152033", "5E6878", "E7ECF3", "D8EAE7", "AA4D43"
SERIF, SANS = "Georgia", "Arial"
W, H = 13.333, 7.5
LM, CW = 0.67, 12.0  # left margin and content width (inches)


def rgb(h):
    return RGBColor.from_string(h)


class Deck:
    def __init__(self, total):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(W), Inches(H)
        self.total, self.n = total, 0

    # ---- primitives
    def new(self, eyebrow, title, notes=""):
        self.n += 1
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        bg = s.background.fill
        bg.solid()
        bg.fore_color.rgb = rgb(PAPER)
        if eyebrow:
            self.text(s, eyebrow.upper(), LM, .42, CW, .3, 12, TEAL, bold=True)
        if title:
            self.text(s, title, LM, .72, CW, .75, 28, NAVY, bold=True, font=SERIF)
        self.text(s, f"{self.n} / {self.total}", W - 2.0, H - .55, 1.4, .3, 11, MUTED, align=PP_ALIGN.RIGHT)
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        return s

    def text(self, s, txt, x, y, w, h, size=15, color=INK, bold=False, font=SANS, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, italic=False, spacing=1.12):
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = anchor
        self._fill(tf, txt, size, color, bold, font, align, italic, spacing)
        return tb

    @staticmethod
    def _runs(p, txt, size, color, bold, font, italic):
        """Text with **bold** segments."""
        parts = txt.split("**")
        for i, seg in enumerate(parts):
            if not seg:
                continue
            r = p.add_run()
            r.text = seg
            r.font.size, r.font.name, r.font.italic = Pt(size), font, italic
            r.font.bold = bold or (i % 2 == 1)
            r.font.color.rgb = rgb(color)

    def _fill(self, tf, txt, size, color, bold, font, align, italic=False, spacing=1.12):
        lines = txt if isinstance(txt, list) else [txt]
        for k, line in enumerate(lines):
            p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
            p.alignment = align
            p.line_spacing = spacing
            if k:
                p.space_before = Pt(size * .45)
            bullet = isinstance(line, str) and line.startswith("• ")
            self._runs(p, line, size, color, bold, font, italic)

    def box(self, s, x, y, w, h, fill=CARD, radius=.06):
        shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
        shp.adjustments[0] = radius
        shp.fill.solid()
        shp.fill.fore_color.rgb = rgb(fill)
        shp.line.fill.background()
        shp.shadow.inherit = False
        return shp

    def card(self, s, x, y, w, h, label=None, head=None, body=None, fill=CARD, size=14.5, big=None, big_color=RED):
        self.box(s, x, y, w, h, fill)
        cy = y + .2
        if label:
            self.text(s, label.upper(), x + .25, cy, w - .5, .3, 11.5, TEAL, bold=True)
            cy += .36
        if head:
            self.text(s, head, x + .25, cy, w - .5, .5, 17, NAVY, bold=True, font=SERIF)
            cy += .5
        if big:
            self.text(s, big, x + .25, cy, w - .5, .6, 30, big_color, bold=True, font=SERIF)
            cy += .65
        if body:
            self.text(s, body, x + .25, cy, w - .5, y + h - cy - .15, size, INK)

    def take(self, s, txt, y, h=None, x=LM, w=CW, size=15.5):
        if h is None:
            h = .62 if len(txt) <= 92 else .82
        if h > .7 and y + h > 6.5:
            y = 6.5 - h
        self.box(s, x, y, w, h, NAVY, .12)
        self.text(s, txt, x + .3, y, w - .6, h, size, PAPER, anchor=MSO_ANCHOR.MIDDLE)

    def foot(self, s, txt):
        self.text(s, txt, LM, H - .55, 10.6, .35, 11, MUTED)

    def image(self, s, path, x, y, w=None, h=None):
        im = Image.open(path)
        ar = im.width / im.height
        if w and h:
            if w / h > ar:
                w = h * ar
            else:
                h = w / ar
        elif w:
            h = w / ar
        else:
            w = h * ar
        return s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))

    def table(self, s, rows, x, y, w, col_w=None, size=13, row_h=.36, header_fill=MIST, bad=(), new_cols=(), bold_first=True):
        """rows[0] = header. Cells may contain 'RED:' prefix to colour red."""
        nr, nc = len(rows), len(rows[0])
        gt = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
        tbl = gt.table
        tbl.first_row = True
        col_w = col_w or [w / nc] * nc
        for j, cw in enumerate(col_w):
            tbl.columns[j].width = Inches(cw)
        for i in range(nr):
            tbl.rows[i].height = Inches(row_h)
            for j in range(nc):
                c = tbl.cell(i, j)
                c.fill.solid()
                c.fill.fore_color.rgb = rgb(header_fill if i == 0 else (CARD if i % 2 == 0 else PAPER))
                c.margin_left = c.margin_right = Inches(.08)
                c.margin_top = c.margin_bottom = Inches(.03)
                c.vertical_anchor = MSO_ANCHOR.MIDDLE
                val = str(rows[i][j])
                color = NAVY if i == 0 else INK
                if j in new_cols and i > 0:
                    color = TEAL
                tf = c.text_frame
                tf.word_wrap = True
                p = tf.paragraphs[0]
                p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
                segs = val.split("RED:")
                for k, seg in enumerate(segs):
                    if not seg:
                        continue
                    if k > 0:
                        num, _, rest = seg.partition(" ")
                        for t, col in ((num, RED), ((" " + rest) if rest else "", color)):
                            if t:
                                r = p.add_run(); r.text = t
                                r.font.size, r.font.name, r.font.color.rgb = Pt(size), SANS, rgb(col)
                                r.font.bold = (col == RED) or i == 0 or (j == 0 and bold_first)
                    else:
                        r = p.add_run(); r.text = seg
                        r.font.size, r.font.name, r.font.color.rgb = Pt(size), SANS, rgb(color)
                        r.font.bold = i == 0 or (j == 0 and bold_first) or (j in new_cols)
        return gt

    def save(self, path):
        self.prs.save(path)
        print(path)


# ====================================================================== shared slides (used by both decks)
def slide_setting(d, notes_extra=""):
    s = d.new("Research setting", "Are Treasury yield loadings stable over time?",
              "Data: JKV = Jungbacker, Koopman and van der Wel (2014) CRSP unsmoothed Fama–Bliss zero yields, 17 maturities, "
              "1970–2009 (licensed, not redistributed). Treasury8 = official par curve, 8 maturities; equal to H.15 CMT in 1990–2009. "
              "Residual autocorrelation: results/serial/residual_acf_summary.csv and bandwidth_real_grid.csv (rank 3, fixed loadings). "
              "Figure: figures/01_treasury_yields.png. " + notes_extra)
    d.image(s, FIG / "01_treasury_yields.png", LM, 1.55, w=7.3)
    d.image(s, ASSET / "eq_model.png", LM + .2, 4.35, h=.33)
    x = LM + 7.6
    d.card(s, x, 1.5, 4.4, 1.25, label="JKV 17", body="CRSP / Fama–Bliss unsmoothed zero-coupon yields, 17 maturities (3 months–10 years), monthly 1970–2009.", size=12.5)
    d.card(s, x, 2.85, 4.4, 1.25, label="Treasury 8", body="Official par yields (H.15 CMT), 8 maturities (3 months–10 years), monthly 1990–2026.", size=12.5)
    d.card(s, x, 4.2, 4.4, 1.3, label="The problem", fill=PALE,
           body="Fixed three-factor residuals are persistent: median lag-1 autocorrelation **0.63** (JKV) and **0.73** (Treasury).", size=13.5)
    d.take(s, "Persistent noise can look like moving loadings, so the test must hold its 5% size before any rejection means a break.", 5.65)


def slide_literature(d):
    s = d.new("Literature", "Where this work sits",
              "Su and Wang (2017) JoE 198(1) 84–101; Su and Wang (2020) ET 36(6); Breitung and Eickmeier (2011) JoE 163(1); "
              "Chen, Dolado and Gonzalo (2014) JoE 180(1); Han and Inoue (2015) ET 31(5); Breitung and Tenhofen (2011) JASA 106(495); "
              "Kreiss (1992); Bühlmann (1997) Bernoulli 3(2); Shao (2010) JASA 105(489); Kiefer and Vogelsang (2005) ET 21(6); "
              "Shao and Zhang (2010) JASA 105(491); Litterman and Scheinkman (1991) JFI; Diebold and Li (2006) JoE; Duffee (2011) RFS 24(9); "
              "Jungbacker, Koopman and van der Wel (2014) JAE 29(1); Bauer and Hamilton (2018) RFS 31(2).")
    cols = [
        ("Testing for changing loadings", [
            "**Su & Wang (2017, JoE):** kernel-local vs fixed PCA loadings; smooth and abrupt change. Our base test.",
            "**Su & Wang (2020, ET):** regression form of the same idea.",
            "**Breitung & Eickmeier (2011); Chen, Dolado & Gonzalo (2014); Han & Inoue (2015):** break tests at a date; little power against smooth change."]),
        ("Handling serial correlation", [
            "**Breitung & Tenhofen (2011, JASA):** GLS for factor models by AR quasi-differencing. Our filter.",
            "**Kreiss (1992); Bühlmann (1997):** sieve-AR bootstrap with a data-chosen order. Our calibration.",
            "**Shao (2010)** dependent wild bootstrap; **Kiefer & Vogelsang (2005)** fixed-b; **Shao & Zhang (2010)** self-normalisation: checked or set aside."]),
        ("Yield-curve factors", [
            "**Litterman & Scheinkman (1991):** level, slope, curvature.",
            "**Diebold & Li (2006):** dynamic Nelson–Siegel loadings.",
            "**Duffee (2011, RFS):** a hidden fourth factor.",
            "**Jungbacker, Koopman & van der Wel (2014):** the JKV data.",
            "**Bauer & Hamilton (2018, RFS):** persistence distorts size in bond tests."]),
    ]
    for k, (lab, items) in enumerate(cols):
        x = LM + k * 4.07
        d.card(s, x, 1.55, 3.86, 3.95, label=lab, body=items, size=12.5)
    d.take(s, "Gap: Su–Wang’s local-vs-fixed test has no version that stays valid when errors are serially correlated, the normal case for yield residuals.", 5.7)


def slide_diagnosis(d):
    s = d.new("The original problem", "What we tested, and why it rejects",
              "Su–Wang (2017) feasible J with Epanechnikov kernel, boundary kernel, rule-of-thumb h0 = (2.35/sqrt 12) T^(-1/5) N^(-1/10); "
              "null p-value from the Section 4.6 Gaussian bootstrap, independent over months. Size: results/serial/ar_bootstrap_study_summary.csv "
              "(paper_iid): JKV 3.5 / 95 / 100%, Treasury 3.5 / 62.5 / 100% for IID / AR(0.3) / AR(0.6). Su–Wang DGP 1 with AR errors: "
              "results/serial/suwang_dgp_replication.csv. MATLAB check: results/serial/matlab_equivalence.csv.")
    d.image(s, ASSET / "eq_J.png", LM, 1.55, h=.5)
    d.image(s, ASSET / "eq_mds.png", LM + 7.7, 1.6, h=.32)
    d.text(s, "B̂ and V̂ assume unpredictable errors.", LM + 7.7, 1.98, 4.2, .3, 13, MUTED)
    w = 3.86
    d.card(s, LM, 2.5, w, 2.95, label="What we did", size=13.5,
           body="Ran Su–Wang’s full-sample J test on yields with 3 and 4 factors: kernel-local PCA loadings vs fixed ones, p-value from the paper’s Gaussian bootstrap, which draws errors independently month by month.")
    d.card(s, LM + 4.07, 2.5, w, 2.95, label="Why it rejects", big="95 / 100%", size=13.5,
           body="False rejections (JKV) with AR(0.3) / AR(0.6) noise and no change. Persistent noise makes local loadings drift; the independent-draw bootstrap never produces that drift.")
    d.card(s, LM + 8.14, 2.5, w, 2.95, label="Relation to the paper", size=13.5,
           body="Same statistic and bootstrap; our code matches the authors’ MATLAB to 12 digits. The theory assumes no serial correlation, and none of the paper’s 10 designs has it. Adding AR(0.3) to its own DGP 1: size 2% → 88%.")
    d.take(s, "So a rejection on yields is not yet evidence of a loading break: the test is used outside the conditions the paper proves it for.", 5.7)


def slide_method(d):
    s = d.new("New method", "Sieve-AR GLS-J: filter the errors, then compare",
              "Implementation: code/serial_robust.py, gls_j_test(p='aic', hmult=2). (1) PCA factors; one AR order p in 1..6 by pooled AIC, "
              "fixed in the bootstrap; quasi-difference yields and factors (Breitung–Tenhofen 2011). (2) Local loading by kernel-weighted LS on "
              "filtered data, h = 2 h0; full-sample loading by OLS. (3) Wild bootstrap: one Rademacher sign per month shared across maturities, "
              "errors rebuilt by the AR recursion, full re-estimation; B = 199 in simulations, 999 on data.")
    d.card(s, LM, 1.45, CW, 1.15, label="The idea", fill=PALE, size=14,
           body="Su–Wang’s comparison is valid when noise is unpredictable. So strip the predictable part out of each maturity’s noise with an AR filter, compare local and fixed loadings on the filtered data, and judge the gap with a bootstrap that puts the persistence back.")
    w = 3.86
    specs = [("1 · Filter", "Fit an AR(p) to each maturity’s residual, one order p chosen by AIC; quasi-difference yields and factors.", "eq_ar.png"),
             ("2 · Compare", "Kernel-local vs full-sample loadings on the filtered data, as in Su–Wang (bandwidth 2h₀, fixed in advance).", "eq_M.png"),
             ("3 · Calibrate", "Flip the sign of each month’s filtered residuals at random, rebuild noise through the fitted AR, refit, recompute. p = share of draws above the data.", "eq_boot.png")]
    for k, (lab, body, eq) in enumerate(specs):
        x = LM + k * 4.07
        d.card(s, x, 2.75, w, 2.4, label=lab, body=body, size=13)
        d.image(s, ASSET / eq, x + .25, 4.45, w=w - .5, h=.55)
    d.card(s, LM, 5.3, 5.9, 1.1, label="Kept from Su–Wang", size=13.5, body="The null λᵢ(t) = λᵢ, the local-vs-fixed comparison, the kernel, and the 3- and 4-factor tests.")
    d.card(s, LM + 6.1, 5.3, 5.9, 1.1, label="New", size=13.5, body="Noise is whitened first, the AR order comes from the data, and the bootstrap carries the dynamics.")


def slide_bandwidth(d):
    s = d.new("Bandwidth", "Tuning h first, with the number of factors fixed",
              "Advisor feedback: if Su–Wang residuals stay strongly serially correlated, either h is too large or there are too few factors; "
              "with the rank fixed at 3 or 4, tune h before testing. Real data: results/serial/bandwidth_real_grid.csv (rank 3; JKV 1970–2009, "
              "CMT 1981–2026). Simulation: bandwidth_original_test_summary.csv (200 panels, T = 240, B = 199). New-test row: AR(1)-filtered "
              "GLS-J at 0.5, 1, 2 h0 (250 panels; simulation_all_runs_summary.csv, gls_h05 / gls_M_wild / gls_h2); the sieve version picks order 1 here. "
              "The two power columns use different alternatives; the like-for-like comparison is the next slide.")
    d.text(s, "With r fixed at 3, h = c·h₀ around Su–Wang’s rule h₀. Narrower h lowers residual autocorrelation, but the original test then over-rejects more; the new test holds size at every h (we fix h = 2h₀).",
           LM, 1.42, CW, .6, 14, MUTED)
    d.text(s, "REAL DATA · MEDIAN LAG-1 AUTOCORRELATION OF RESIDUALS", LM, 2.08, CW, .3, 11.5, TEAL, bold=True)
    d.table(s, [["Loadings", "Fixed", "Local, c = 2", "Local, c = 1", "Local, c = 0.5", "Local, c = 0.25"],
                ["JKV", "0.63", "0.54", "0.45", "0.36", "0.22"],
                ["Treasury", "0.73", "0.69", "0.62", "0.52", "0.36"]], LM, 2.38, CW,
            [2.6] + [1.88] * 5, size=13, row_h=.34)
    d.text(s, "SIMULATION · EACH CELL = SIZE % / POWER %", LM, 3.55, CW, .3, 11.5, TEAL, bold=True)
    d.table(s, [["Test · true noise", "JKV c = 0.5", "JKV c = 1", "JKV c = 2", "Treas. c = 0.5", "Treas. c = 1", "Treas. c = 2"],
                ["Original · independent", "3.5 / 51", "3.5 / 46.5", "2.5 / 40.5", "2.5 / 54.5", "0.5 / 36.5", "0 / 26.5"],
                ["Original · AR(0.3)", "RED:98.5 / 100", "RED:92 / 99.5", "RED:62.5 / 93", "RED:89.5 / 98.5", "RED:60 / 88.5", "RED:28.5 / 65"],
                ["New GLS-J · AR(0.6)", "2.8 / 17.6", "2.4 / 19.2", "2.4 / 20.8", "4.0 / 30.4", "1.6 / 34.0", "1.2 / 44.4"]],
            LM, 3.85, CW, [2.82] + [1.53] * 6, size=13, row_h=.36, new_cols=())
    d.text(s, "Size: loadings constant, want ≈ 5. Power, original test: a 4th factor appears mid-sample at half strength; new test: loadings shift by half the calibrated scale. Red: size far above 5%, so the power next to it is mostly false alarms.",
           LM, 5.4, CW, .6, 12, MUTED)


def slide_sizepower(d):
    s = d.new("Size and power", "The new test keeps size and gains power",
              "All cells: T = 240, 5% level, B = 199; same calibration and the same change in every column (half the calibrated scale, "
              "loadings shift from mid-sample). Original and + AR bootstrap: 200 panels (results/serial/ar_bootstrap_study_summary.csv; "
              "paper_iid, ar1_global). New: sieve-AR GLS-J, AIC order, h = 2h0, 250 panels (simulation_all_runs_summary.csv, gls_aic_h2). "
              "Heterogeneous AR 0.2–0.7: size 100/97 original, 3.5/3.5 AR bootstrap, 1.6/3.2 new; new-test power 28.0/47.2.")
    d.text(s, "Simulations calibrated to each yield panel. **Each cell = size % / power %**: size wants ≈ 5, power wants high.", LM, 1.42, CW, .4, 14, MUTED)
    d.table(s, [["True noise", "Original · JKV", "Original · Treasury", "+ AR boot · JKV", "+ AR boot · Treasury", "New GLS-J · JKV", "New GLS-J · Treasury"],
                ["Independent", "3.5 / 44", "3.5 / 72.5", "3.5 / 39", "3.5 / 68", "2.4 / 49.6", "1.2 / 76.8"],
                ["AR(0.3)", "RED:95 / 99.5", "RED:62.5 / 97", "3.0 / 18", "4.5 / 35", "1.6 / 26.4", "1.2 / 55.2"],
                ["AR(0.6)", "RED:100 / 100", "RED:100 / 100", "1.0 / 7.5", "1.5 / 23", "2.4 / 20.8", "2.0 / 44.4"]],
            LM, 1.95, CW, [1.86] + [1.69] * 6, size=15, row_h=.62, new_cols=(5, 6))
    d.text(s, "**Size**: loadings constant. **Power**: loadings shift mid-sample by half the calibrated scale. Red: size far above 5%; the power next to it is mostly false alarms. **Original**: Su–Wang J, bootstrap noise independent over months. **+ AR boot**: same J, bootstrap noise follows AR(1).",
           LM, 4.6, CW, .8, 12.5, MUTED)
    d.take(s, "The AR bootstrap fixes size but loses power; the new test keeps size under every AR design and detects more changes.", 5.65)


def slide_next(d, page_question=True):
    s = d.new("Limits and next steps", "One open limit, a first real-data read, and the plan",
              "Limit: results/serial/final_method_size_power.csv (null_volslope_ar03/ar06: after T/2 the noise SD multiplier rises linearly "
              "from 1 at the short end to 2 at the long end, loadings fixed). Quick checks (100 panels): T-growing AR cap changes nothing; "
              "two-half weighted PCA raises size to 21–84%. Real data: results/serial/empirical_frozen_pmax6.csv (JKV, licensed panel) and "
              "empirical_treasury8_public.csv (rerun through Aug 2026). Not pre-registered.")
    d.card(s, LM, 1.5, 3.9, 3.95, label="Remaining limit", size=13, body=[
        "If noise volatility rises unevenly across maturities (short ×1, long ×2), size is 14–15% (JKV) and 8–10% (Treasury).",
        "Two quick fixes failed: a T-growing AR order changes nothing; two-half weighted PCA is worse (up to 84%).",
        "In 1990–2009 the residual SD ratio, 2nd vs 1st half, is 1.0–1.9 (JKV) and 1.05–1.5 (Treasury)."])
    d.box(s, LM + 4.08, 1.5, 4.25, 3.95)
    d.text(s, "REAL DATA · BOOTSTRAP p-VALUES", LM + 4.3, 1.7, 3.9, .3, 11.5, TEAL, bold=True)
    d.table(s, [["Sample", "r = 3", "r = 4"], ["JKV 1990–2009", "0.026", "0.196"], ["Treasury 1990–2009", "0.005", "0.350"],
                ["Treasury 1990–2026", "0.003", "0.108"], ["JKV 1970–89", "0.007", "0.001"]],
            LM + 4.3, 2.05, 3.85, [2.05, .9, .9], size=13, row_h=.4)
    d.text(s, "Exploratory, B = 999. 1970–89 moves to ≈ 0.05 if the AR order may reach 12.", LM + 4.3, 4.2, 3.85, .7, 12, MUTED)
    d.card(s, LM + 8.5, 1.5, 3.5, 3.95, label="Next steps", fill=PALE, size=13, body=[
        "1. Lock the specification: samples, r = 3 then 4, AIC order rule, h = 2h₀, B = 999.",
        "2. Date the break: track where the local-vs-fixed gap builds up over t.",
        "3. Theory: bootstrap validity with fixed N, T → ∞."])
    d.take(s, "Question for you: can we assume maturity-specific noise volatility is stable within each test window, or should we split windows at known regime changes?", 5.65, h=.75)


# ====================================================================== main deck
def main_deck():
    d = Deck(24)
    # 1 title
    s = d.new("", "", "Main storyline deck. Sources: figures/ and results/; reproducible with python code/run.py figures and the notebook.")
    d.box(s, 0, 0, W, H, NAVY, 0)
    d.text(s, "TREASURY YIELD-CURVE FACTOR STABILITY · RESEARCH UPDATE", LM, 1.6, CW, .4, 13, "9FD8CF", bold=True)
    d.text(s, "Are Treasury yield loadings time-varying?", LM, 2.1, CW, .8, 34, PAPER, bold=True, font=SERIF)
    d.text(s, "From the Su–Wang benchmark to a test that stays valid with persistent errors", LM, 3.05, CW, .6, 20, "D8DEE8", font=SERIF, italic=True)
    d.text(s, ["Benchmark under independent errors  →  why yields break it  →  bandwidth, AR bootstraps, identification",
               "→  Su–Wang audit  →  sieve-AR GLS-J  →  size, power, first real-data read  →  open questions"], LM, 4.3, CW, 1.0, 15, "D8DEE8")
    d.text(s, "Working draft, November 2026. All numbers are simulation or exploratory results unless stated.", LM, 6.3, CW, .4, 12, "AEB8C8")

    # 2 answer first
    s = d.new("Summary", "The answer so far, in four lines",
              "Each line is backed by a later slide: benchmark (6–8), the problem (10–14), the new test (16–20), real data (21).")
    rows = [("1", "Under independent errors the Su–Wang test works on yields:", "5% size 3.8%, power 100% for a strong fourth factor."),
            ("2", "Yield residuals are persistent (lag-1 ≈ 0.6–0.7), and then it fails:", "AR(0.3) noise with no change is rejected 62–95% of the time, in Su–Wang’s own DGP too."),
            ("3", "Sieve-AR GLS-J keeps the Su–Wang null and comparison, and repairs size:", "0.8–7.6% in 27 of 32 null cells, with more power than an AR bootstrap."),
            ("4", "First real-data read (exploratory):", "constant three-factor loadings rejected in 1990–2009; four-factor not rejected.")]
    for k, (n, a, b) in enumerate(rows):
        y = 1.6 + k * 1.0
        d.box(s, LM, y, .55, .55, TEAL, .5)
        d.text(s, n, LM, y, .55, .55, 18, PAPER, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        d.text(s, [a, b], LM + .8, y - .03, CW - .8, .9, 15.5, INK)
    d.take(s, "Open: if noise volatility changes unevenly across maturities, size rises to 8–15%. Can we assume it is stable within a test window?", 5.7)

    slide_setting(d)
    slide_literature(d)

    # 5 Su-Wang test
    s = d.new("The base test", "Su–Wang (2017): compare local and fixed loadings",
              "Equations from Su and Wang (2017), Section 4. Implementation: code/sw_test.py (equation-based, checked against the authors' MATLAB; "
              "see results/serial/matlab_equivalence.csv). Bandwidth rule h0 = (2.35/sqrt 12) T^(-1/5) N^(-1/10).")
    d.image(s, ASSET / "eq_model.png", LM, 1.6, h=.36)
    d.image(s, ASSET / "eq_J.png", LM, 2.2, h=.5)
    w = 3.86
    d.card(s, LM, 3.0, w, 2.35, label="Local fit", size=13.5, body="At each month t, PCA on kernel-weighted covariance of nearby months (Epanechnikov kernel, boundary-corrected, h = rule of thumb).")
    d.card(s, LM + 4.07, 3.0, w, 2.35, label="Fixed fit", size=13.5, body="One PCA on the whole sample. M̂ is the mean squared gap between the two common components; B̂ and V̂ centre and scale it.")
    d.card(s, LM + 8.14, 3.0, w, 2.35, label="p-value", size=13.5, body="Paper’s bootstrap (Section 4.6): fixed common component plus Gaussian errors with the residual covariance, drawn independently month by month.")
    d.take(s, "Why this test: it targets smooth and abrupt loading change without assuming a shape for λᵢ(t).", 5.7)

    # 6 benchmark design
    s = d.new("Benchmark · design", "Calibrated fake curves with a known answer",
              "figures/02_simulation_design.png. JKV calibration (calibration/model_jkv.npz): Nelson–Siegel loadings, AR(1) factors; "
              "fourth factor from the Svensson fit. T = 480, N = 17. Seeds: tune for exploring, final used once.")
    d.image(s, FIG / "02_simulation_design.png", LM, 1.5, w=8.0)
    d.card(s, LM + 8.25, 1.5, 3.75, 3.95, label="Design", size=13.5, body=[
        "No change: three Nelson–Siegel factors, fixed loadings, independent errors.",
        "Change: a fourth (Svensson) factor switches on at month 160, 240 or 320; full or half strength.",
        "Size = rejections with no change; power = rejections with the change."])
    d.take(s, "This alternative tests whether three factors stay adequate; a fixed-rank loading shift is tested separately later.", 5.7)

    # 7 benchmark results
    s = d.new("Benchmark · result", "With independent errors, size is fine and power is high",
              "results/final_tables.md Table G (500 final-seed panels) and Table C (300 tune panels). figures/03 and 04.")
    d.image(s, FIG / "03_fullsample_size_power.png", LM, 1.5, w=6.3)
    d.image(s, FIG / "04_bandwidth.png", LM + 6.5, 1.5, w=5.5)
    d.table(s, [["Full-sample test", "1%", "5%", "10%"], ["Size, no change", "1.2", "3.8", "7.6"],
                ["Power, full strength, τ = 160/240/320", "100 / 100 / 99.8", "100 / 100 / 100", "100 / 100 / 100"],
                ["Power, half strength, τ = 160/240/320", "76.6 / 80.2 / 68.4", "91.6 / 91.4 / 82.4", "95.8 / 94.2 / 89.0"]],
            LM, 3.95, CW, [3.9, 2.7, 2.7, 2.7], size=13, row_h=.38)
    d.take(s, "Narrower bandwidth trades a little size for power; the paper’s rule (c = 1) is acceptable and slightly conservative.", 5.7)

    # 8 dating
    s = d.new("Benchmark · dating", "Detecting a change is not dating it",
              "results/final_tables.md Tables B and G; figures/06_dating.png. Scan: month-by-month windows, first rejection = date. "
              "Gated split: full-sample test first, then least-squares split (Bai 1997; Baltagi, Kao and Wang 2017).")
    d.image(s, FIG / "06_dating.png", LM, 1.5, w=7.6)
    d.card(s, LM + 7.85, 1.5, 4.15, 3.95, label="What we learned", size=13.5, body=[
        "Sequential scan: procedure size 7–13% at nominal 1%; strong breaks dated about 20 months late.",
        "Test, then split: size = the test’s size; strong middle/late breaks within ±24 months in 80–87% of panels.",
        "Dating is only as good as the first-stage test, so the rest of the work fixes the test."])
    d.take(s, "Dating is secondary: it needs a valid first-stage test, which persistent errors take away.", 5.7)

    # 9 original empirical
    s = d.new("First real-data result", "The original test rejects constant loadings",
              "paper/working_draft.tex Table 'Full-sample tests of constant loadings' (B = 9,999, rule-of-thumb h).")
    d.table(s, [["Panel", "d = 3", "d = 4", "d = 5"],
                ["JKV 1970–2009 (N = 17, T = 480): Ĵ / 99% crit.", "91.0 / 24.7", "65.1 / 86.8", "79.3 / 92.1"],
                ["   p-value", "≤ 0.0001", "0.342", "0.247"],
                ["Treasury 1981–2026 (N = 8, T = 540): Ĵ / 99% crit.", "108.8 / 6.9", "92.2 / 19.9", "117.1 / 99.8"],
                ["   p-value", "≤ 0.0001", "≤ 0.0001", "0.0012"]], LM, 1.6, CW, [5.4, 2.2, 2.2, 2.2], size=14, row_h=.5)
    d.card(s, LM, 4.3, CW, 1.15, label="But", fill=PALE, size=14,
           body="These p-values come from a bootstrap that draws errors independently over months. Before reading them as loading change, we need to know what the test does when the errors are persistent.")
    d.take(s, "Strong rejections, but only as reliable as the independent-error assumption behind the bootstrap.", 5.7)

    # 10 persistence
    s = d.new("The problem · data", "Real residuals are strongly persistent",
              "results/serial/residual_acf_summary.csv (standardised PCA; JKV 1970–2009, CMT 1981–2026). All maturities have positive lag-1 autocorrelation.")
    d.table(s, [["Panel", "Fixed loadings, r = 3", "Local fit, r = 3", "Fixed loadings, r = 4", "Local fit, r = 4"],
                ["JKV 17, 1970–2009", "0.63", "0.45", "0.49", "0.33"],
                ["Treasury CMT 8, 1981–2026", "0.73", "0.62", "0.55", "0.47"]], LM, 1.6, CW, [3.6, 2.1, 2.1, 2.1, 2.1], size=14, row_h=.5)
    d.card(s, LM, 3.35, 5.9, 2.1, label="What it means", size=13.5,
           body="Month-to-month noise in each maturity is predictable from last month. A run of high noise looks like a temporary new level of that maturity, which the local fit can absorb as a loading change.")
    d.card(s, LM + 6.1, 3.35, 5.9, 2.1, label="What the bootstrap assumes", size=13.5,
           body="Errors are independent across months (Assumption A.7(i), martingale difference). Its null distribution has no such runs, so the observed gap looks extreme.")
    d.take(s, "The question becomes: how badly does persistence alone mislead the test?", 5.7)

    # 11 stress test
    s = d.new("The problem · simulation", "Persistence alone makes the original test reject",
              "results/final_tables.md Table H (gated split at 5%, 200 tune panels, τ = 240); figures/05_serial_dependence.png.")
    d.image(s, FIG / "05_serial_dependence.png", LM, 1.5, w=8.0)
    d.card(s, LM + 8.25, 1.5, 3.75, 3.95, label="Reading", size=13.5, big="100%", body=[
        "No-change panels rejected once the error AR coefficient is 0.3 or more.",
        "AR(1) bootstrap on global residuals: size fine, power collapses. On local residuals: power back, size 7.5–24.5%."])
    d.take(s, "Neither first AR(1) bootstrap is acceptable, so we went back to the design of the test.", 5.7)

    # 12 AR bootstraps
    s = d.new("Fix attempt 1", "AR-aware bootstraps fix size but lose power",
              "results/serial/ar_bootstrap_study_summary.csv: T = 240, 200 panels, B = 199; power = half-strength loading shift.")
    d.table(s, [["Method", "Independent · JKV", "Independent · Treas.", "AR(0.3) · JKV", "AR(0.3) · Treas.", "AR(0.6) · JKV", "AR(0.6) · Treas."],
                ["Original", "3.5 / 44", "3.5 / 72.5", "RED:95 / 99.5", "RED:62.5 / 97", "RED:100 / 100", "RED:100 / 100"],
                ["AR(1) bootstrap, global residual", "3.5 / 39", "3.5 / 68", "3.0 / 18", "4.5 / 35", "1.0 / 7.5", "1.5 / 23"],
                ["AR(1) bootstrap, local residual", "RED:11 / –", "6.0 / –", "RED:11 / –", "9.0 / –", "RED:14.5 / –", "RED:16 / –"]],
            LM, 1.6, CW, [3.0] + [1.5] * 6, size=13, row_h=.55)
    d.text(s, "Each cell = size % / power %. Red: size well above 5%. Local-residual power was not run on the loading-shift design.", LM, 3.9, CW, .4, 12, MUTED)
    d.card(s, LM, 4.3, CW, 1.25, label="Why", size=13.5,
           body="The AR coefficient is estimated from residuals that still contain any loading change, so under the alternative the bootstrap noise becomes too persistent and swallows the signal.")
    d.take(s, "We need a test whose statistic, not only its bootstrap, accounts for persistence.", 5.7)

    slide_bandwidth(d)  # 13

    # 14 identification
    s = d.new("Identification limit", "Some error changes look exactly like a new factor",
              "results/serial/shape_test_key_results.csv (500 panels unless noted). The shape test assumes the first 120 months are stable and asks "
              "whether a fourth Nelson–Siegel-type curvature appears later.")
    d.image(s, ASSET / "eq_ident.png", LM, 1.6, h=.3)
    d.card(s, LM, 2.2, 5.9, 3.25, label="Consequence", size=13.5, body=[
        "If the error covariance may change arbitrarily, “a fourth common factor appears” and “the errors gain a common component” are observationally equivalent.",
        "Any test must therefore assume something about how the noise can change."])
    d.box(s, LM + 6.1, 2.2, 5.9, 3.25)
    d.text(s, "A CONDITIONAL SHAPE TEST (REJECTION %)", LM + 6.3, 2.4, 5.5, .3, 11.5, TEAL, bold=True)
    d.table(s, [["Case", "JKV", "Treasury"], ["No change, AR(0.3)", "4.0", "5.4"], ["No change, AR(0.6)", "4.8", "5.6"],
                ["Half-strength 4th factor, AR(0.3)", "90.4", "72.6"], ["Noise ×1.5 after month 120, no change", "RED:99", "RED:99"]],
            LM + 6.3, 2.75, 5.5, [3.3, 1.1, 1.1], size=13, row_h=.45)
    d.take(s, "This is why the new test assumes noise volatility changes, if any, in the same proportion across maturities.", 5.7)

    # 15 Su-Wang audit
    s = d.new("Back to Su–Wang", "The failure is in the method, not in our code",
              "results/serial/matlab_equivalence.csv (Octave run of the authors' test_common_component.m on the same matrices); "
              "results/serial/suwang_dgp_replication.csv (N = T = 100, R = 2, 200 panels, B = 199); figures/08_suwang_dgp_serial.png.")
    d.card(s, LM, 1.5, 4.7, 3.95, label="Implementation check", size=13, body=[
        "SM1 and SM2 equal the authors’ MATLAB to 12 significant digits on JKV and on Su–Wang DGP 1.",
        "Finding: the paper’s B̂ and V̂ depend on the sign convention of local eigenvectors (Ĵ = 60.9 vs 71.3 on the same JKV matrix). Procrustes alignment recommended."])
    d.image(s, FIG / "08_suwang_dgp_serial.png", LM + 4.9, 1.5, w=7.1)
    d.take(s, "In Su–Wang’s own design, AR(0.3) errors raise size from 2% to 88%: the over-rejection is the method, not our data or code.", 5.7)

    slide_method(d)  # 16
    slide_sizepower(d)  # 17

    # 18 robust size
    s = d.new("Robustness of size", "Size holds across 16 error designs, with one exception",
              "figures/09_sieve_gls_size.png; results/serial/final_method_size_power.csv and simulation_all_runs_summary.csv (gls_aic_h2), "
              "250 panels each; Monte Carlo 95% band around 5% is about 2.3–7.7%.")
    d.image(s, FIG / "09_sieve_gls_size.png", LM, 1.45, h=4.1)
    d.card(s, LM + 7.2, 1.5, 4.8, 3.95, label="Reading", size=13, body=[
        "Holds: AR(0.3–0.9), AR by maturity, AR(2), ARMA(1,1), persistent + measurement noise, cross-maturity correlation (constant or switching on), volatility ×1.5 at every maturity, rank 4.",
        "Fails: volatility rising unevenly across maturities (8–15%), the identification limit of slide 14.",
        "JKV rank 4 with IID errors: 10.4% (weak fourth factor)."])
    d.take(s, "27 of 32 null cells between 0.8% and 7.6%.", 5.7)

    # 19 power and choices
    s = d.new("Power and design choices", "Why AIC, why h = 2h₀, and how power grows",
              "figures/10_sieve_gls_power.png; ablations in results/serial/simulation_all_runs_summary.csv (same seeds): gls_M_wild (AR(1), h0), "
              "gls_sieve (BIC), gls_aic_h2 (final), gls_h2, swqd_J_r, dwb_J_l12.")
    d.image(s, FIG / "10_sieve_gls_power.png", LM + .5, 1.4, w=11.0)
    d.table(s, [["Variant (JKV / Treasury, %)", "ARMA(1,1) size", "Meas. noise size", "AR(0.6) power, half"],
                ["Fixed AR(1) filter", "RED:73.6 / 47.6", "RED:100 / 98", "19.2 / 34.0"],
                ["Sieve, BIC order", "RED:13.2 / 16.8", "RED:55.6 / 34.4", "19.2 / 33.6"],
                ["Sieve, AIC order, h = 2h₀ (final)", "4.0 / 3.6", "7.6 / 4.8", "20.8 / 44.4"]],
            LM, 5.0, CW, [4.2, 2.6, 2.6, 2.6], size=12.5, row_h=.32)


    # 20 open limit
    s = d.new("Open limit", "The open case: uneven volatility change",
              "final_method_size_power.csv (null_volslope_ar03/06); quick checks in simulation_all_runs_summary.csv run quick8 "
              "(gls_aicT_h2, gls_split_h2; 100 panels). SD ratios: fixed three-factor residuals, 1990–2009.")
    d.table(s, [["No-change design (size %)", "JKV", "Treasury"], ["Volatility ×1.5 at all maturities, AR(0.3)", "2.0", "2.8"],
                ["Volatility ×1 (short) → ×2 (long), AR(0.3)", "RED:14.4", "RED:8.4"], ["Volatility ×1 (short) → ×2 (long), AR(0.6)", "RED:15.2", "RED:10.4"],
                ["Same, AR(0.6), T-growing AR order (pmax = 14)", "RED:16", "RED:9"], ["Same, AR(0.6), two-half weighted PCA", "RED:57", "RED:21"]],
            LM, 1.6, 7.3, [4.9, 1.2, 1.2], size=13, row_h=.48)
    d.card(s, LM + 7.5, 1.6, 4.5, 3.85, label="Interpretation", size=13, body=[
        "Uneven volatility change rotates the PCA factor space, which the statistic reads as a loading change.",
        "Real 1990–2009 residual SD ratio (2nd / 1st half): 1.03–1.92 JKV, 1.05–1.48 Treasury, the same order as the simulated case.",
        "So r = 3 rejections cannot be attributed to loadings alone."])
    d.take(s, "We state stable relative noise volatility within a window as an assumption, and report the SD ratios with every test.", 5.7)

    # 21 empirical
    s = d.new("Real data", "Three factors rejected, four not, in 1990–2009",
              "figures/11_empirical_pvalues.png; JKV: results/serial/empirical_frozen_pmax6.csv / pmax12.csv (licensed panel). Treasury: "
              "results/serial/empirical_treasury8_public.csv (committed H.15 file through Aug 2026; the exploratory run also had a partial Sep 2026 row). "
              "B = 999, AIC order, h = 2h0. Not pre-registered.")
    d.image(s, FIG / "11_empirical_pvalues.png", LM, 1.45, w=7.3)
    d.card(s, LM + 7.5, 1.5, 4.5, 3.95, label="Reading", size=13, body=[
        "1990–2009: r = 3 rejected (0.026 JKV, 0.005 Treasury); r = 4 not (0.196, 0.350). Part of the three-factor “loading change” may be a missing fourth factor (Duffee 2011).",
        "1970–89 is sensitive to the AR order cap (p ≈ 0.05 at pmax = 12).",
        "2010–2026 rejects at both ranks but sits on the zero lower bound: the open-limit case, not interpreted."])
    d.take(s, "Exploratory: the formal run comes after locking samples, rank order, AR rule, h and B.", 5.7)

    # 22 contributions
    s = d.new("Contribution", "What is new in this work",
              "Comparisons: results/serial/*.csv; earlier candidates: AR bootstraps (slide 12), bandwidth (13), shape test (14).")
    items = [("1", "A Su–Wang-type test that holds size with persistent errors", "Same null and local-vs-fixed comparison; AR filter with a data-chosen order and a sieve bootstrap."),
             ("2", "Robust to two kinds of error-covariance change", "Proportional volatility change and cross-maturity correlation change: size 2.0–4.8%, where the shape test fails."),
             ("3", "A quantified identification boundary", "Uneven volatility change gives 8–15% size: a concrete version of the factor-vs-error equivalence."),
             ("4", "Two informative negative results", "Dependent wild bootstrap on the original J (size 12–50%); quasi-differenced original J (no power on yields)."),
             ("5", "Implementation audit of Su–Wang", "MATLAB equivalence, sign-convention dependence of B̂ and V̂, failure in the paper’s own DGP.")]
    for k, (n, a, b) in enumerate(items):
        y = 1.5 + k * .82
        d.box(s, LM, y, .5, .5, TEAL, .5)
        d.text(s, n, LM, y, .5, .5, 16, PAPER, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        d.text(s, [f"**{a}**", b], LM + .75, y - .04, CW - .75, .8, 14, INK)

    # 23 limits & next
    s = d.new("Limits and plan", "What we cannot claim yet, and the next three steps",
              "Limits are listed in results/key_findings.md and notebook/research_notebook.ipynb, Section 13.")
    d.card(s, LM, 1.5, 5.9, 3.95, label="Not yet", size=13.5, body=[
        "No asymptotic theory: bootstrap validity with fixed N, PCA factors and T → ∞ is open.",
        "Uneven volatility change: size 8–15%.",
        "Weak loading change (0.25 scale): power 6–8%.",
        "JKV r = 4 with IID errors: size 10.4%.",
        "Empirical results not pre-registered."])
    d.card(s, LM + 6.1, 1.5, 5.9, 3.95, label="Next", fill=PALE, size=13.5, body=[
        "1. Lock the specification: samples (1990–2009 main), r = 3 then 4, AIC order rule, h = 2h₀, B = 999; run once.",
        "2. Date the break: the statistic is a sum over t, so its time profile with a bootstrap band locates where the gap builds up.",
        "3. Theory draft: sieve-bootstrap consistency for the GLS statistic (Kreiss 1992; Bühlmann 1997; Breitung–Tenhofen 2011)."])
    d.take(s, "Question for you: can we assume maturity-specific noise volatility is stable within each test window, or split windows at known regime changes?", 5.7, h=.75)

    # 24 reproducibility
    s = d.new("Appendix", "Where everything is",
              "Repository layout. The notebook is the main working record; the paper is an AI-written draft.")
    d.table(s, [["Item", "Location"],
                ["Main working record (figures, tables, conclusions)", "notebook/research_notebook.ipynb"],
                ["Original Su–Wang test and week-9 benchmark", "code/sw_test.py, code/simulation.py, code/dating.py"],
                ["New sieve-AR GLS-J test, serial-error DGPs, empirical", "code/serial_robust.py"],
                ["Figures 01–11", "python code/run.py figures  →  figures/"],
                ["Frozen week-9 tables / new result tables", "results/final_tables.md  ·  results/serial/"],
                ["Port check vs exploratory code (40 panels, identical decisions)", "results/serial/port_equivalence.csv"],
                ["Paper draft", "paper/working_draft.pdf"]], LM, 1.6, CW, [6.0, 6.0], size=13.5, row_h=.48)
    d.save(Path(__file__).resolve().parent / "Research_Update_Full_Storyline.pptx")


# ====================================================================== weekly deck (current seven-page update, unchanged storyline)
def weekly_deck():
    d = Deck(7)
    slide_setting(d)
    slide_literature(d)
    slide_diagnosis(d)
    slide_method(d)
    slide_bandwidth(d)
    slide_sizepower(d)
    slide_next(d)
    d.save(Path(__file__).resolve().parent / "Weekly_Quick_Take_2026-11-25.pptx")


if __name__ == "__main__":
    main_deck()
    weekly_deck()
