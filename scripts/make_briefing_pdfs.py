"""Build the two team-briefing PDFs (full + quick) from the repo's result CSVs.

Needs reportlab, matplotlib, pandas (NOT in requirements.txt; install in any venv).
    python scripts/make_briefing_pdfs.py
Output: docs/briefing/SafeCharge_Full_Briefing.pdf, docs/briefing/SafeCharge_Quick_Guide.pdf
Numbers in tables/charts are read from results/; numbers in text come from results/FINDINGS_phase1-3.md and docs/PROJECT_LOG.md.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TAB = RES / "tables"
OUT = ROOT / "docs" / "briefing"
FIG = OUT / "fig"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(exist_ok=True)

# colour-blind-safe (Okabe-Ito), one colour per method everywhere
COL = {"PPOLag": "#D55E00", "EDF": "#0072B2", "LLF": "#009E73", "MPC": "#CC79A7", "FCFS": "#8A8A8A", "OPT": "#222222"}
NAME = {"PPOLag": "Learned (PPO-Lag)", "EDF": "EDF", "LLF": "LLF", "MPC": "MPC", "FCFS": "FCFS", "OPT": "Offline optimum"}
INK = "#222222"
ACCENT = colors.HexColor("#1F4E79")
LIGHT = colors.HexColor("#EAF1F8")
GREY = colors.HexColor("#F3F3F3")


# ----------------------------------------------------------------------------- charts
def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def fig_e1():
    d = pd.read_csv(TAB / "e1_m0_E1_long.csv")
    d = d[d.scenario == "test_M0"].set_index("method")
    order = ["OPT", "MPC", "FCFS", "PPOLag", "EDF", "LLF"]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.7))
    for ax, (c, lo, hi, ttl, k) in zip(axs, [("cost_per_kwh", "cost_per_kwh_lo", "cost_per_kwh_hi", "Cost per kWh ($), lower is better", 1),
                                              ("unmet_share", "unmet_share_lo", "unmet_share_hi", "Unmet energy (%), lower is better", 100)]):
        v = d.loc[order, c] * k
        err = [(d.loc[order, c] - d.loc[order, lo]) * k, (d.loc[order, hi] - d.loc[order, c]) * k]
        ax.barh(range(len(order)), v, color=[COL[m] for m in order], height=0.6, xerr=err, error_kw=dict(lw=0.8, ecolor="#555555"))
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([NAME[m] for m in order], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(ttl, fontsize=8.5, loc="left")
        ax.grid(axis="x", color="#DDDDDD", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8)
        for i, x in enumerate(v):
            ax.text(x + (err[1].iloc[i]) + v.max() * 0.015, i, f"{x:.3f}" if k == 1 else f"{x:.2f}", va="center", fontsize=7.5, color=INK)
        ax.set_xlim(0, v.max() * 1.25)
    axs[0].set_xlim(0.14, 0.19)
    fig.tight_layout()
    fig.savefig(FIG / "e1.png", dpi=200)
    plt.close(fig)


def fig_e2():
    d = pd.read_csv(TAB / "e1_m0_E2_long.csv").set_index(["scenario", "method"])
    e1 = pd.read_csv(TAB / "e1_m0_E1_long.csv")
    e1 = e1[e1.scenario == "test_M0"].set_index("method")
    ws = [0, 0.5, 1, 2]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for m in ["PPOLag", "EDF", "LLF"]:
        y = [e1.loc[m, "unmet_share"] * 100] + [d.loc[(f"test_M2_w{w}", m), "unmet_share"] * 100 for w in ["0.5", "1", "2"]]
        ax.plot(ws, y, marker="o", ms=4, lw=1.8, color=COL[m])
        ax.text(2.04, y[-1], NAME[m], fontsize=8, color=INK, va="center")
    ax.text(0.0, 3.9, "MPC is off this chart: 0.67% -> 8.9% -> 13.2% -> 20.2%", fontsize=7.5, color=COL["MPC"], va="top")
    ax.set_ylim(0, 4)
    style_ax(ax)
    ax.set_xlim(-0.05, 3.1)
    ax.set_xticks(ws)
    ax.set_xlabel("noise in reported departure, w (hours; 0 = truthful)", fontsize=8)
    ax.set_ylabel("unmet energy (%)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "e2.png", dpi=200)
    plt.close(fig)


def fig_e3():
    d = pd.read_csv(TAB / "e1_m0_E3_pivot_gain_pct.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8), sharey=False)
    for ax, col, ttl in zip(axs, ["paired_bgM0", "paired_bgM0_k0.15"], ["Normal grid cap (kappa 0.20)", "Tight grid cap (kappa 0.15)"]):
        for m in ["PPOLag", "EDF", "LLF", "MPC"]:
            x = d[d.method == m]
            ax.plot(x.s, x[col], marker="o", ms=4, lw=1.8, color=COL[m], label=NAME[m])
        style_ax(ax)
        ax.set_title(ttl, fontsize=8.5, loc="left")
        ax.set_xticks([0.5, 1, 2])
        ax.set_xlabel("how many hours earlier the driver reports (s)", fontsize=8)
    axs[0].set_ylabel("extra energy gained (% of request)", fontsize=8)
    axs[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "e3.png", dpi=200)
    plt.close(fig)


def fig_train():
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.7))
    for sd in range(5):
        L = pd.read_csv(RES / "runs" / "e1_m0" / f"seed{sd}" / "log.csv")
        L = L[L.val_U.notna()]
        axs[0].plot(L["update"], L.val_U * 100, lw=1.0, color="#D55E00", alpha=0.55)
        axs[1].plot(L["update"], L.val_cost_per_kwh, lw=1.0, color="#D55E00", alpha=0.55)
    axs[0].axhline(2, color="#444444", lw=1, ls="--")
    axs[0].text(1500, 2.1, "2% budget", fontsize=7.5, ha="right", va="bottom")
    axs[0].set_title("Unmet energy on validation days (%)", fontsize=8.5, loc="left")
    axs[1].set_title("Cost per kWh on validation days ($)", fontsize=8.5, loc="left")
    for ax in axs:
        style_ax(ax)
        ax.set_xlabel("PPO update (5 seeds, one line each)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "train.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- tables from CSV
def e1_rows():
    d = pd.read_csv(TAB / "e1_m0_E1_long.csv")
    d = d[d.scenario == "test_M0"].set_index("method")
    opt = d.loc["OPT", "cost_per_kwh"]
    rows = [["Method", "Cost $/kWh", "Gap to optimum", "Unmet energy", "Missed-deadline vehicles", "High-power ratio"]]
    for m in ["OPT", "MPC", "FCFS", "PPOLag", "EDF", "LLF"]:
        r = d.loc[m]
        gap = "-" if m == "OPT" else f"+{r.gap_cost_per_kwh * 100:.1f}%"
        rows.append([NAME[m], f"{r.cost_per_kwh:.4f}", gap, f"{r.unmet_share * 100:.2f}%", f"{r.miss_rate * 100:.1f}%", f"{r.high_power_ratio:.2f}"])
    return rows


def e2_rows():
    d = pd.read_csv(TAB / "e1_m0_E2_long.csv").set_index(["scenario", "method"])
    e1 = pd.read_csv(TAB / "e1_m0_E1_long.csv")
    e1 = e1[e1.scenario == "test_M0"].set_index("method")
    sc = [("Truthful (M0)", None), ("Noisy w=0.5 h", "test_M2_w0.5"), ("Noisy w=1 h", "test_M2_w1"), ("Noisy w=2 h", "test_M2_w2"),
          ("Late, 50% of cars (M3)", "test_M3_w2_q0.5"), ("Recorded real reports (M1r)", "test_M1r")]
    ms = ["PPOLag", "EDF", "LLF", "MPC"]
    rows = [["Unmet energy (%)"] + [NAME[m] for m in ms]]
    for lab, s in sc:
        rows.append([lab] + [f"{(e1.loc[m, 'unmet_share'] if s is None else d.loc[(s, m), 'unmet_share']) * 100:.2f}" for m in ms])
    return rows


def e3_rows():
    g = pd.read_csv(TAB / "e1_m0_E3_pivot_gain_pct.csv")
    sh = pd.read_csv(TAB / "e1_m0_E3_pivot_share_gain_gt0.csv")
    rows = [["Method", "Gain at s=0.5 h", "s=1 h", "s=2 h", "Share of drivers who gain"]]
    for m in ["PPOLag", "EDF", "LLF", "MPC"]:
        x = g[g.method == m].sort_values("s")
        y = sh[sh.method == m]
        rows.append([NAME[m]] + [f"+{v:.2f}%" for v in x.paired_bgM0] + [f"{y.paired_bgM0.min() * 100:.0f}-{y.paired_bgM0.max() * 100:.0f}%"])
    return rows


# ----------------------------------------------------------------------------- pdf helpers
ss = getSampleStyleSheet()
BASE = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=9.6, leading=13.2, textColor=INK, alignment=TA_LEFT, spaceAfter=4)
SMALL = ParagraphStyle("s", parent=BASE, fontSize=8.2, leading=10.8, textColor=colors.HexColor("#555555"))
H1 = ParagraphStyle("h1", parent=BASE, fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=ACCENT, spaceBefore=6, spaceAfter=6, keepWithNext=1)
H2 = ParagraphStyle("h2", parent=BASE, fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=ACCENT, spaceBefore=8, spaceAfter=3, keepWithNext=1)
CELL = ParagraphStyle("c", parent=BASE, fontSize=8.4, leading=10.6, spaceAfter=0)
CELLB = ParagraphStyle("cb", parent=CELL, fontName="Helvetica-Bold", textColor=colors.white)
TITLE = ParagraphStyle("t", parent=H1, fontSize=24, leading=28, spaceAfter=4)


def P(t, st=BASE):
    return Paragraph(t, st)


def B(items, st=BASE):
    return ListFlowable([ListItem(Paragraph(i, st), leftIndent=10, bulletColor=ACCENT) for i in items], bulletType="bullet", start="-", leftIndent=12, bulletFontSize=9)


def T(rows, widths, head=True, zebra=True, align_right_from=None):
    data = [[Paragraph(str(c), CELLB if (head and r == 0) else CELL) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if head else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if head:
        st.append(("BACKGROUND", (0, 0), (-1, 0), ACCENT))
    if zebra:
        for i in range(1, len(rows)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), GREY))
    t.setStyle(TableStyle(st))
    return t


def BOX(text, bg=LIGHT):
    t = Table([[Paragraph(text, BASE)]], colWidths=[174 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("BOX", (0, 0), (-1, -1), 0.6, ACCENT),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t


def IMG(path, width_mm):
    from PIL import Image as PI
    w, h = PI.open(path).size
    return Image(str(path), width=width_mm * mm, height=width_mm * mm * h / w)


def build(path, story, title):
    def foot(c, d):
        c.saveState()
        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.HexColor("#777777"))
        c.drawString(18 * mm, 10 * mm, f"SafeCharge-RL  |  Team 32  |  {title}")
        c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {d.page}")
        c.restoreState()
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=18 * mm,
                      title=title, author="Team 32").build(story, onFirstPage=foot, onLaterPages=foot)


def cover(sub):
    return [Spacer(1, 10 * mm), P("SafeCharge-RL", TITLE),
            P("Reinforcement learning for EV charging when drivers' departure times are wrong or misreported", ParagraphStyle("x", parent=H2, fontSize=12.5, leading=16)),
            P(sub, BASE), Spacer(1, 3 * mm),
            T([["Team 32", "Adithyan R (AM.SC.U4AIE23065), Adithya Rajesh (AM.SC.U4AIE23007), Balu M Krishna (AM.SC.U4AIE23026)"],
               ["Guide", "Dr. Geetha M, Dept. of CSE (AI), Amrita Vishwa Vidyapeetham"],
               ["Prepared", "9 Oct 2026, for the mid-review (9-10 Oct). Code: github.com/adithyx-0/safecharge-rl (private repo)"]], [25, 149], head=False, zebra=False),
            Spacer(1, 4 * mm)]


# ----------------------------------------------------------------------------- shared content
def projection_example():
    return T([["Illustrative example (made-up numbers, not data)", ""],
              ["Grid cap", "20 kW"],
              ["Policy asks for (3 cars)", "12 kW, 10 kW, 8 kW  (total 30 kW, too much)"],
              ["Safety layer subtracts the same amount mu from each", "mu = 3.33 kW"],
              ["Power actually given", "8.67 kW, 6.67 kW, 4.67 kW  (total 20 kW, safe)"]], [80, 94])


# ============================================================================= FULL
def full():
    s = cover("Full briefing: everything about the project, from zero. Written so a teammate who has never seen it can follow.")
    s += [BOX("<b>Read this first (one paragraph).</b> Many EVs plug in at a shared charging station. Someone must decide, every 15 minutes, how much power each car gets, "
              "without exceeding the station's power limit, while keeping the electricity bill low and getting cars charged before they leave. "
              "The scheduler only knows the departure time the <b>driver typed in</b>, and drivers often type wrong values, or could type early times on purpose to get charged first. "
              "We train an RL policy on real charging data and ask: <b>how badly does it suffer when reports are wrong, and can a driver cheat it?</b>"),
          Spacer(1, 3 * mm)]

    s += [P("Contents", H2), B(["1. The problem in simple words", "2. Why reinforcement learning", "3. What others did and the gap", "4. Our research question and hypotheses",
                                "5. The data and how we prepared it", "6. The RL formulation (MDP)", "7. Algorithm and why", "8. System design and code map",
                                "9. Baselines and metrics", "10. Experiments E1-E6", "11. Results so far", "12. What is novel", "13. Progress, what is left, honest caveats",
                                "14. Likely questions and answers", "15. Slide-by-slide plan for the mid-review", "16. Miscellaneous: glossary, how to run, numbers to remember"])]
    s.append(PageBreak())

    # 1
    s += [P("1. The problem in simple words", H1),
          P("<b>Setting.</b> A workplace car park with N charging ports (54 at Caltech, 52 at JPL). Cars arrive at random times, need some energy (kWh), and leave at some time. "
            "The whole car park sits behind one transformer, so the total power of all cars together must never go above a <b>grid limit</b>."),
          P("<b>Three things pull in different directions:</b>"),
          B(["<b>Cost:</b> electricity price changes through the day (cheap at night, expensive 12:00-18:00), so we prefer to charge when it is cheap.",
             "<b>Deadlines:</b> each driver needs the car charged before leaving, so we cannot just wait for cheap hours.",
             "<b>Battery wear:</b> charging at very high power is harder on the battery. We have no battery data, so we use a simple proxy: how often a port runs above 80% of its maximum power."]),
          P("<b>The twist (our contribution).</b> The scheduler does not see when the car really leaves. It only sees the departure time the driver <i>reported</i>. "
            "Real drivers get this wrong (on real data the average error is about 3 hours). A driver could also report an earlier time on purpose, hoping to be served first. "
            "The car always leaves at its real time, whatever was reported."),
          P("<b>Our research question.</b> When reported departure times are wrong or deliberately early, how much do cost and missed deadlines get worse for a learned policy compared with simple rules, "
            "how much extra charge can a cheating driver get, and can a small change in the policy or training reduce that?"),
          P("<b>Problem statement in one line.</b> Learn a charging policy that never exceeds the grid limit, keeps unmet energy within a small budget (2% of requested energy), "
            "keeps battery wear within a budget, and minimises electricity cost, when it only sees reported (possibly wrong) departure times.")]

    # 2
    s += [P("2. Why reinforcement learning", H1),
          B(["<b>Decisions are made one step at a time without knowing the future.</b> Exact optimisation (linear programming) needs all future arrivals, departures and prices. We only use it as an after-the-fact yardstick (the 'offline optimum').",
             "<b>Simple rules look at one thing only.</b> First-Come-First-Served ignores urgency; Earliest-Deadline-First ignores price. A learned policy can weigh price, urgency, fairness and wear together.",
             "<b>MPC (re-solving an optimisation every step) is a fair rival,</b> good when forecasts and reports are accurate, but slower per step and sensitive to wrong inputs. We include it as a baseline.",
             "<b>RL has no built-in safety guarantee,</b> so we add a hard safety layer (section 6) and constraint budgets (Lagrangian)."])]

    # 3
    s += [P("3. What others did, the gap, and how we differ", H1),
          T([["Paper", "What it does", "What is missing"],
             ["Li, Wan, He (2020), IEEE Trans. Smart Grid", "Safe deep RL for charging as a constrained problem", "One charger only; no shared grid limit; safety is learned, not guaranteed"],
             ["Chen, Shi (2022)", "Augmented Lagrangian + SAC for EV charging", "One EV, synthetic arrivals; no hard-constraint layer"],
             ["Ting et al. (2024)", "Multi-agent RL with a station power limit; early departures", "Simulated EVs; departure uncertainty as random noise; no comparison with simple rules"],
             ["Li, Sun (2023)", "Out-of-distribution-aware charging on Caltech data; shows user-entered departure times are unreliable", "Small station (2 chargers); no deadlines/prices; not about misreporting"],
             ["Ferragut et al. (2021)", "EDF/LLF with uncertain departures; under-reporting drivers get more service; fix: serve only until the declared departure", "Rule-based policies only (no learning), no prices"],
             ["Gabriele et al. (2026)", "Forecast the missing departure time, trained with the controller", "Departure missing, not wrong or strategic"],
             ["Lee, Li, Low (2019)", "ACN-Data: the real dataset we use", "Not an RL paper"]], [38, 70, 66]),
          Spacer(1, 2 * mm),
          P("<b>Products:</b> ChargePoint power management (configured limits and schedules), Enel X JuiceNet (closed in N. America Oct 2024), Caltech ACN (uses MPC with driver-entered times), OCPP (a message standard, does not say how to compute limits). None is a published learned policy."),
          BOX("<b>The gap.</b> To our search, nobody treats the reported departure time as something a driver may deliberately misreport to a <i>learned</i> policy, tested on <i>real</i> sessions with a <i>shared hard grid limit</i>, "
              "compared with rules and an offline-optimal schedule. That is what SafeCharge-RL studies. We do <b>not</b> claim a new RL algorithm, and we do not claim 'safety projection + Lagrangian' is new by itself.")]

    # 4
    s += [P("4. Hypotheses", H1),
          T([["H1", "A policy trained on reported departures gives more charge to drivers who report earlier ones (the Ferragut effect, now for a learned policy)."],
             ["H2", "Cost and missed deadlines get worse as the noise level, or the share of misreporting drivers, grows."],
             ["H3", "Serving a car only until its declared departure, and randomising reports during training, reduce the misreporting gain, at some efficiency cost."]], [14, 160], head=False, zebra=False),
          P("<b>Limit we state openly:</b> misreporting is simulated, so we make no claim about how often real drivers lie. Data are workplace/campus sites, not home or public charging.", SMALL)]

    # 5
    s += [P("5. The data and how we prepared it", H1),
          P("<b>ACN-Data</b> (Caltech's Adaptive Charging Network): real charging sessions, free after registering for an API token. Downloaded and verified: <b>Caltech 31,424 sessions, JPL 33,638</b> (equal to the API's own totals, no duplicates, no zero-energy sessions). "
            "The token lives in an environment variable / <font face='Courier'>.env</font> and is never committed; raw data are gitignored."),
          T([["Session field", "Used as"], ["connectionTime", "Arrival time (rounded up to the 15-min grid)"], ["disconnectTime", "TRUE departure (rounded down); the car always leaves here"],
             ["kWhDelivered", "Energy the car needs, e_i (clipped to what one port can deliver in the stay)"], ["siteID", "Caltech (train/val/test) vs JPL (transfer test)"]], [45, 129]),
          P("Constants from the data: each port gives at most <b>6.656 kW</b> (32 A x 208 V). Price: SCE time-of-use tariff, summer weekday, $/kWh 0.05623 (00-08h), 0.0925 (08-12h), "
            "<b>0.26668 (12-18h, peak)</b>, 0.0925 (18-23h), 0.05623 (23-24h).", BASE),
          P("Choices we made while preparing the data (all in <font face='Courier'>configs/base.yaml</font>, logged in <font face='Courier'>docs/PROJECT_LOG.md</font>)", H2),
          T([["Decision", "What we did and why"],
             ["Episode", "One weekday = 96 steps of 15 minutes."],
             ["Midnight-crossing sessions", "Dropped (6.3% at Caltech, 1.5% at JPL). Clipping them would invent an earlier departure, which is exactly what we study."],
             ["Caltech fast station", "Excluded: one station delivers about 45 kW, not 6.656 kW (3.6% of sessions, 8.6% of energy). Replay shortfall vs recorded energy falls from 7.4% to 1.2%."],
             ["Tariff", "Summer price curve applied to ALL weekdays (a controlled assumption). Real Jun-Sep days are too few (about 85 in our period) and mixing seasons would add a price shift."],
             ["Daylight saving", "No weekday affected (clock changes fall on Sundays)."],
             ["Simulator", "Own small numpy/Gymnasium environment (fast, about 32,000 steps/s on one core) instead of ACN-Sim, which works in amps and its data client failed authentication. The design allowed this as its fallback."],
             ["Grid cap", "P_grid = kappa x P_peak. Chosen kappa = 0.20 (P_grid 34.6 kW): the tightest value where the 2% unmet budget is still achievable with full information. Also tested 0.15 (too tight) and 0.30 (loose). JPL uses 0.28 (95 kW)."]], [40, 134]),
          P("Splits (chronological, one pricing regime, before COVID)", H2),
          T([["Split", "Dates (local)", "Days", "Vehicles"], ["Train", "2019-01-01 to 2019-08-31", "174", "6,206"], ["Validation", "2019-09-01 to 2019-10-31", "44", "1,536"],
             ["Test", "2019-11-01 to 2020-02-29", "85", "2,720"], ["Post-shift test", "2021-01-01 to 2021-09-30 (after COVID)", "182", "only 9 per day (vs 34)"], ["JPL transfer", "same test / post windows", "85 / 176", "5,329 / 5,533"]], [30, 70, 24, 50]),
          P("Surprising finding about reports", H2),
          P("The submitted text says only 0.2% of sessions have a driver-entered departure time. <b>That is wrong for the full data</b> (the 500-session sample was all from April 2018, before the app collected it): "
            "<b>52% of Caltech and 94% of JPL sessions have one.</b> Real drivers are far from accurate: mean error 3.15 h (Caltech) and 2.35 h (JPL); only about 35-39% are within 1 hour; about 60% say a time earlier than the real one. "
            "We kept the design's modes unchanged and added the recorded reports as an extra mode, <b>M1r</b> (supplementary). The team must correct the 0.2% statement in the report."),
          P("Mode sizes: the design's noise w up to 2 h gives mean error w/2 = 1 h at most, much milder than reality.", SMALL)]

    # 6
    s += [P("6. The RL formulation (MDP)", H1),
          T([["Part", "Definition (plain words)"],
             ["Agent", "The charging controller of the station."],
             ["Environment", "The station simulator: ports, vehicles from real sessions, price, grid limit. Built by us in numpy."],
             ["Episode / end", "One weekday of 96 steps (15 min). Ends after step 96. A car's unmet energy is counted when it actually leaves."],
             ["State (4N+4 numbers)", "Per port: occupied?, energy still needed r, time left until REPORTED departure tau, laxity = tau - r / P_port (spare time if charged at full rate; negative = cannot finish). "
                                      "Global: price, sin and cos of time of day, last step's total power / grid limit. Empty ports are zero and masked. <b>The true departure time is never shown.</b>"],
             ["Action", "A number u between 0 and 1 for each port = the fraction of that port's maximum useful power it asks for."],
             ["Reward", "Minus the electricity cost of the energy delivered this step (scaled to about unit size)."],
             ["Cost 1: unmet energy", "When a car leaves: energy it did not get / reference energy (8.36 kWh). Budget: 2% of requested energy per day."],
             ["Cost 2: battery wear proxy", "Share of ports running above 80% of max power at each step. Budget set from the rule-based baselines (2.92 per day)."],
             ["Safety projection", "Hard grid limit: see below. It lives inside the environment step, so the limit can never be broken."],
             ["Objective", "Maximise reward subject to both cost budgets (a constrained MDP). Discount 0.99 in the design."]], [38, 136]),
          P("The safety projection in one picture", H2),
          P("If the policy's requests add up to more than the grid limit, the layer subtracts the <b>same amount mu</b> from every request (never below 0) until the total fits. "
            "mu is found by a quick bisection (about 20 tries). If the request already fits, nothing changes. This is the exact closest feasible point, and our tests check it against a general QP solver."),
          projection_example(), Spacer(1, 2 * mm),
          P("Reporting modes (how the reported time is made from the true one)", H2),
          T([["Mode", "Meaning"], ["M0", "Truthful: reported = true departure."], ["M2 (noisy)", "reported = true + random error uniform in [-w, +w], w = 0.5, 1, 2 hours."],
             ["M3 (late)", "A share q of drivers report later than the truth by up to w hours."], ["M4 (strategic)", "A share rho (0.1, 0.25, 0.5) report s hours EARLIER, s = 0.5, 1, 2."],
             ["M1r (extra)", "Real recorded reports from ACN-Data where they exist, else a resampled real error."], ["Note", "There is no 'M1' on purpose (the design dropped it). Reports are never earlier than arrival. All methods get the same random draws."]], [30, 144])]

    # 7
    s += [P("7. Algorithm and why", H1),
          B(["<b>PPO-Lagrangian.</b> PPO is a standard, stable policy-gradient method. The Lagrangian part adds one multiplier (lambda) per cost; lambda goes up when a cost is over budget and down when under (dual ascent). This turns 'stay within budget' into a trade-off the policy learns. SAC-Lagrangian was 'if time allows' (not done).",
             "<b>Policy network: Deep Sets.</b> One small shared network per port, average over occupied ports to get a 'station context', then a shared head gives each port's request. Sharing parameters means the same policy works for any number of cars and any port order (tested).",
             "<b>Critics</b> (reward, cost 1, cost 2) use the same shared structure.",
             "<b>Memoryless:</b> the observation is treated as the state (true departure is hidden, so this is only approximate).",
             "<b>Our own single-file PPO-Lagrangian (about 170 lines)</b> instead of a library: a library (OmniSafe) was checked but pulls about 40 extra packages and, to our knowledge, supports one cost, while we need two plus a custom network. This differs from the design wording 'existing implementation'."]),
          P("Additions that are NOT in the submitted design (must be disclosed)", H2),
          B(["<b>Behaviour-cloning warm start from LLF:</b> from scratch the policy stayed worse than every rule after 2,000 updates (unmet energy arrives only at departure, so learning is slow). We first copy LLF's behaviour, then let RL improve.",
             "Discount 0.999 instead of 0.99; dual-step clipping and a cap on lambda (lambda once wound up to 89); learning-rate decay.",
             "Count of occupied ports added to the pooled context (the average alone cannot tell how many cars compete); policy outputs a logit passed through a sigmoid.",
             "Training: 5 seeds x 1,500 updates each, tuned on validation days only (seed 0); test days used only for reported numbers."])]

    # 8
    s += [P("8. System design and code map", H1),
          IMG(ROOT / "docs" / "architecture_diagram.jpg", 174),
          P("Flow: real sessions -> report model (truthful or misreported) -> environment (ports, price, grid limit) -> RL agent proposes power per port -> safety projection enforces the limit -> applied power steps the environment and returns the next state and reward.", SMALL),
          T([["File", "What it does"], ["src/safecharge/data.py", "Loads sessions, local time, 15-min rounding, splits, exclusions"], ["report_model.py", "M0, M2, M3, M4 (and M1r) with seeded draws"],
             ["env.py", "Gymnasium environment; projection inside step; optional curtail mode"], ["projection.py", "Bisection projection (uniform cut; priority variant for E5)"],
             ["baselines.py / mpc.py / offline_lp.py", "FCFS, EDF, LLF / receding-horizon LP / hindsight-optimal LP"], ["metrics.py", "All metrics, bootstrap CIs, paired replay"],
             ["agents/deepsets.py, ppo_lag.py", "The policy network and the PPO-Lagrangian trainer"], ["scripts/", "train.py, run_experiment.py, run_paired_replay.py, make_tables.py, data download/audit/calibration"],
             ["tests/", "108 tests, all passing (run on 9 Oct 2026)"], ["results/", "CSV tables; FINDINGS_phase1-3.md has the plain bullet findings"]], [58, 116])]

    # 9
    s += [P("9. Baselines and metrics", H1),
          T([["Baseline", "Idea", "Sees"], ["FCFS", "Serve in arrival order up to the grid limit", "ignores reports"], ["EDF", "Earliest reported departure first", "reported time"],
             ["LLF", "Least laxity first (most urgent)", "reported time + energy"], ["MPC", "Re-solve an optimisation each step using reported deadlines", "reported time"],
             ["Offline optimum", "Linear program with the TRUE departures and the whole day known; a lower bound (checked on every test day)", "true time (cheating)"],
             ["Unprojected RL", "RL with a soft grid penalty instead of projection (planned, not built)", "-"]], [32, 100, 42]),
          P("<b>Metrics:</b> cost per kWh; <b>unmet energy U</b> (share of requested energy not delivered); missed-deadline rate (car got under 95% of its request); grid violations (count and size); "
            "<b>high-power ratio W</b> (battery-wear proxy); Jain fairness; gap to the offline optimum; <b>misreporting gain</b>; decision time per step. Mean over test days with 95% bootstrap confidence intervals; identical days and report draws for every method.", BASE),
          BOX("<b>Paired replay (the misreporting-gain test).</b> Pick one focal car. Run the same day twice with the same policy and the same other reports: once the car reports truthfully, once it reports s hours earlier. "
              "Gain = extra energy it received in the second run (kWh and % of its request). With s = 0 the gain is exactly 0 (tested).")]

    # 10
    s += [P("10. Experiments", H1),
          T([["ID", "Question", "Status"], ["E1", "Learned policy vs baselines, truthful reports", "Done (5 seeds)"], ["E2", "How much worse with noisy (M2) / late (M3) reports?", "Done (5 seeds)"],
             ["E3", "Can a driver gain by reporting early? (paired replay, tests H1, H2)", "Done (5 seeds)"],
             ["E4", "Fixes: (a) serve only until declared departure; (b) train with randomised reports (tests H3)", "(a) done on existing policy; (b) 6 of 9 training runs finished, 3 paused"],
             ["E5", "Projection ablation: uniform cut vs priority-weighted vs none", "Switches built; no test-split run"], ["E6", "Transfer: JPL site and post-COVID period", "Done"]], [14, 110, 50])]

    # 11
    s += [P("11. Results so far", H1),
          P("All numbers are on the 85 test days, kappa 0.20, 5 trained seeds, 0 grid violations for every method. Source: <font face='Courier'>results/tables/</font> and <font face='Courier'>results/FINDINGS_phase1-3.md</font>.", SMALL),
          P("E1: truthful reports", H2), T(e1_rows(), [32, 24, 28, 26, 36, 28]), Spacer(1, 2 * mm), IMG(FIG / "e1.png", 170),
          B(["The learned policy <b>ties EDF</b> on cost and unmet energy. It does <b>not</b> beat the rules; MPC (with truthful deadlines) is the cheapest, 5.5% above the optimum.",
             "Its special trait: <b>gentle charging.</b> High-power ratio 0.17 vs 0.77-0.79 for everything else, so the wear proxy is about 40% lower at similar cost.",
             "Weak spots: 7.4% of cars end below 95% (EDF 1.2%), and short stays (under 2 h) get 3.0% unmet vs 0.0-0.5% for EDF/LLF.",
             "Decision time per step: rules 0.045 ms, learned 0.14 ms, MPC 0.48 ms."]),
          P("Training curves (validation days)", H2), IMG(FIG / "train.png", 170),
          P("The policy starts near LLF (warm start) and then stays near the 2% unmet-energy budget while cost settles. It never reaches the price-shifting savings of MPC or the optimum.", SMALL),
          P("E2: wrong reports make things worse", H2), T(e2_rows(), [60, 28, 28, 28, 28]), Spacer(1, 2 * mm), IMG(FIG / "e2.png", 120),
          B(["Learned policy is about <b>2.6x worse</b> at w = 2 h (1.3% to 3.45%); EDF/LLF about 1.2-1.6x; <b>MPC about 30x worse</b> (it trusts deadlines and postpones charging). MPC only looks cheap because it delivers less energy.",
             "With real recorded reports (M1r) the ranking is the same: LLF 3.0%, EDF 3.8%, learned 7.4%, MPC 19.6%."])]
    s += [P("E3: can a driver cheat? (paired replay, 413 focal cars per cell)", H2), T(e3_rows(), [34, 32, 24, 24, 52]), Spacer(1, 2 * mm), IMG(FIG / "e3.png", 170),
          B(["<b>H1 supported:</b> reporting earlier gives the learned policy's driver extra energy in 93-97% of cases (+0.84% to +1.50% of the request). Rules give a gain only when there is real contention (1.5-3.6% of cases).",
             "<b>H2 supported:</b> gain grows with s and is larger when the cap is tight (learned policy +2.6% to +4.8% at kappa 0.15; LLF is as exploitable as the learned policy there).",
             "In kWh the gain is small (0.03-0.55 kWh per car) because most cars are not in contention."]),
          P("E4 part 1: curtail at declared departure (fix tested on existing policy)", H2),
          B(["With curtailment <b>every method loses by under-reporting</b> (about -0.6% to -1.8% at s up to 1 h, about -9% at s = 2 h): the incentive disappears. With truthful reports nothing changes (as it must).",
             "Price: honest-but-wrong early reports are punished. Unmet energy for noisy w = 2 h: EDF 1.5% to 6.2%, learned 3.5% to 8.3%. Late reports are not hurt."]),
          P("E6: transfer", H2),
          B(["<b>Post-COVID Caltech (M0):</b> fine (unmet 0.31%, rules at most 0.07%).",
             "<b>JPL: the learned policy fails to transfer</b> (unmet 28.6% vs LLF 0.9%), even with a non-binding cap (23.6%). JPL sessions are about 1.7x larger (14.6 vs 8.4 kWh). The reason was not tested directly."]),
          P("Checks that give the numbers credibility", H2),
          B(["Offline-optimum lower bound asserted on every test day for every method and RL seed; 0 grid violations; same days and report draws for all methods; seed spread small (unmet std 0.22 points)."])]

    # 12
    s += [P("12. What is novel (list for slide 6)", H1),
          B(["<b>Reported departure treated as something a driver can misreport</b> to a <i>learned</i> policy, on real multi-EV sessions with a shared hard grid limit.",
             "<b>Misreporting-gain measure by paired replay</b>, and its use on RL, rules and MPC side by side.",
             "<b>Hard safety projection + Lagrangian budgets</b> for deadlines and wear in one multi-vehicle station (the combination is stated as not claimed new by itself; the evaluation setting is the contribution).",
             "<b>Comparison with a hindsight-optimal schedule</b> on real sessions, with the lower bound verified on every day.",
             "<b>Testing the Ferragut fix</b> (serve only until declared departure) and randomised-report training on a learned policy.",
             "<b>Extra finding:</b> real recorded driver reports are common and badly wrong (mean error 2-3 h); the submitted '0.2%' claim does not hold on full data."])]

    # 13
    s += [P("13. Progress, what is left, honest caveats", H1),
          BOX("<b>Progress.</b> In our estimate roughly 80-85% of the planned work is done (not a measured figure). Everything up to E3, E6 and the first part of E4 is finished and reproducible. "
              "The mid-review slide asks for '50% progress'; we are beyond that."),
          Spacer(1, 2 * mm),
          T([["Item", "State"], ["Data download, audit, preprocessing", "Done, verified"], ["Environment, projection, FCFS/EDF/LLF, offline LP, MPC, metrics", "Done, 108 tests pass"],
             ["PPO-Lagrangian, 5 seeds; E1, E2, E3, E6", "Done"], ["E4 part 1 (curtail on standard policy)", "Done"],
             ["E4 part 2 (policies trained with random reports / curtail)", "6 of 9 runs finished; 3 seed-2 runs paused at update 1240, 1280, 760 of 1500. Resume command is at the end of docs/PROJECT_LOG.md (about 1-2 h)"],
             ["E5 projection ablation on test days", "Not done (switches exist)"], ["Unprojected RL baseline", "Not implemented"], ["SAC-Lagrangian", "Not started (optional in design)"],
             ["Report text and slides", "Team to write"]], [78, 96]),
          P("Caveats to remember", H2),
          B(["The learned policy needs a behaviour-cloning warm start that is not in the submitted design. Without it, results are worse than every rule.",
             "Constants we chose (kappa 0.20, JPL 0.28, cost budget d2, M = $5/kWh in the LP, reward scale, splits, tariff on all weekdays) are not in the submitted text; they are logged in docs/PROJECT_LOG.md.",
             "Other-kappa rows (0.15, 0.30) are evaluation-only: the policy was trained at 0.20 and not retrained.",
             "E4 uses 3 seeds (not 5) for time reasons; the randomised-report mixture weights are our choice.",
             "The M1r mode (recorded reports) is supplementary, not part of the design's E1-E6.",
             "The 'why' for the JPL failure and for the policy's gentle charging are plausible explanations, not tested."])]

    # 14
    s += [P("14. Likely questions and short answers", H1),
          T([["Question", "Answer"],
             ["Why does the learned policy not beat the rules?", "Cost comes mostly from when you charge; rules already charge early and fast. The policy matches EDF and charges more gently. It is a real finding; we report it as is."],
             ["Is the grid limit really guaranteed?", "Yes, by construction: the projection is inside the environment step. 0 violations in every run; the projection is tested against a QP solver."],
             ["Why not just use MPC?", "MPC is best when reports are truthful, but collapses (about 30x worse) when they are wrong, and is slower per step."],
             ["Why simulate misreporting?", "Real reports are noisy but we cannot know who lies on purpose. We show sensitivity and exploitability, not how often drivers lie."],
             ["Is the data real?", "Yes, ACN-Data sessions from Caltech and JPL. Price and the grid limit are our assumptions (ACN-Data has no price or battery data)."],
             ["Why is JPL a failure?", "Policy trained on Caltech's smaller sessions; it does not transfer. Honest negative result."],
             ["How do you know results are not luck?", "5 seeds, 85 test days, bootstrap 95% CIs, same days and draws for all methods."]], [60, 114])]

    # 15
    s += [P("15. Slide-by-slide plan for the mid-review", H1),
          T([["Slide", "Put on it", "Where in this PDF"],
             ["1 Title, team", "Title, 3 names, guide, problem area: constrained RL for EV charging under unreliable reports", "Cover"],
             ["2 Problem", "Shared station, grid limit, price, deadlines; reports are wrong or gamed; why sequential RL; one-line problem statement", "1, 2"],
             ["3 Existing work", "Table of 5-6 papers, their limitation, our gap", "3"],
             ["4 MDP + algorithm + data", "State/action/reward/costs/episode; PPO-Lagrangian + Deep Sets, why; data = ACN-Data, our own simulator", "5, 6, 7"],
             ["5 System diagram", "The 5-box figure (docs/architecture_diagram.jpg)", "8"],
             ["6 Novel components", "The list in section 12", "12"],
             ["7 Demo and next plan", "Results chart(s), training curve, status table, remaining work (E4 finish, E5, report)", "11, 13"]], [32, 100, 42]),
          P("Tip: show one result slide with the E3 chart and one with E1; the E3 chart is the central novelty.", SMALL)]

    # 16
    s += [P("16. Miscellaneous", H1),
          P("Glossary", H2),
          T([["Term", "Meaning"], ["EV, kWh, kW", "Electric vehicle; energy (kilowatt-hour); power (kilowatt)"], ["Port / P_port", "One charging plug; its max power, 6.656 kW"],
             ["Grid cap P_grid / kappa", "Limit for total station power = kappa x typical peak (kappa 0.20 is a tight cap)"], ["Laxity", "Spare time = time left minus time needed at full power"],
             ["Reported vs true departure", "What the driver typed vs when the car really leaves"], ["MDP / CMDP", "Decision problem with states, actions, rewards / the same with extra cost budgets"],
             ["PPO", "A stable policy-gradient RL algorithm"], ["Lagrangian / dual ascent", "Turn budgets into weights (lambda) that rise when a budget is broken"], ["Deep Sets", "Network that handles any number of items in any order"],
             ["Projection", "Shrinking a request to the nearest allowed one"], ["FCFS / EDF / LLF", "First-come / earliest-deadline / least-laxity first"], ["MPC", "Re-solving an optimisation every step over a short horizon"],
             ["Offline optimum / LP", "Best schedule if the whole day is known; solved as a linear program"], ["U, W", "Unmet-energy share; high-power ratio (wear proxy)"], ["Curtailment", "Stop charging a car after its declared departure time"],
             ["Bootstrap CI", "Confidence range from resampling test days"], ["Seed", "Random-number start; 5 seeds = 5 independent trainings"]], [46, 128]),
          P("How to run (from the repo root)", H2),
          B(["Setup: Python 3.12 venv, <font face='Courier'>pip install -r requirements.txt</font>; get your own ACN token, set <font face='Courier'>ACN_TOKEN</font> (or <font face='Courier'>.env</font>), run <font face='Courier'>scripts/download_acndata_windowed.py</font> (the older offset-based downloader kept hitting server 502 errors).",
             "Tests: <font face='Courier'>python -m pytest tests -q</font> (108 should pass).",
             "Train one seed: <font face='Courier'>python scripts/train.py configs/e1.yaml --seed 0</font> (about 35-40 min per run on a laptop; resumable).",
             "Evaluate: <font face='Courier'>scripts/run_eval_suite.sh configs/e1.yaml e1_m0</font>, then <font face='Courier'>python scripts/make_tables.py e1_m0</font>.",
             "Trained checkpoints are not in GitHub; teammates must retrain or get them from the owner's laptop.",
             "Never commit <font face='Courier'>.env</font> or <font face='Courier'>data/</font> (both gitignored)."]),
          P("Numbers worth remembering", H2),
          T([["Caltech / JPL sessions", "31,424 / 33,638"], ["Ports", "54 / 52 (Caltech fast station excluded)"], ["Port power", "6.656 kW"], ["Grid cap (main)", "34.6 kW (kappa 0.20)"],
             ["Step / day", "15 min / 96 steps"], ["Peak price", "$0.26668/kWh, 12:00-18:00"], ["Unmet budget d1", "2% of requested energy"], ["Test days", "85 (Caltech, Nov 2019 - Feb 2020)"],
             ["Best cost (optimum)", "$0.1564/kWh; learned $0.1728; MPC $0.1649"], ["Env speed", "about 32,000 steps/s; LP about 3 ms per day"], ["Tests", "108 passing"]], [50, 124], head=False),
          P("Process notes", H2),
          B(["Where to read more: <font face='Courier'>docs/PROJECT_LOG.md</font> (every decision and test), <font face='Courier'>results/FINDINGS_phase1-3.md</font> (plain findings), <font face='Courier'>CLAUDE.md</font> (the implementation brief), <font face='Courier'>docs/final_sections_VIII-XI.md</font> (submitted design).",
             "Server problems seen: ACN API returned transient HTTP 502 errors on deep pages; solved by downloading month by month and checking each month against the API total.",
             "One bug-style incident: a stale run directory was auto-resumed by a new run of the same name; caught from the log and all seeds restarted clean.",
             "The ACN token was once hard-coded in a notebook; it was removed and is now only read from the environment. Rotate it if the project is ever shared publicly."])]
    build(OUT / "SafeCharge_Full_Briefing.pdf", s, "Full briefing")


# ============================================================================= QUICK
def quick():
    s = cover("Quick guide: the whole project on a few pages, in plain words.")
    s += [BOX("<b>One sentence.</b> We train an RL policy to share a charging station's limited power among EVs on real data, and test what happens when drivers' reported departure times are wrong or faked."),
          P("The problem", H2),
          B(["Many cars, one transformer: total power must stay under a <b>grid limit</b>.", "Prices change during the day; every driver has a deadline; high-power charging wears batteries.",
             "The scheduler only sees the departure time the driver <b>reported</b>. Real reports are often wrong (about 3 h off on average), and drivers could report early to be served first."]),
          P("Why RL", H2),
          B(["The future (arrivals, prices) is unknown, so we decide step by step.", "Simple rules look at one thing; RL can balance cost, urgency and wear.", "RL has no built-in safety guarantee, so we add a hard safety layer."]),
          P("What exists and our gap", H2),
          B(["Papers cover constrained RL for one EV, stations with random early departures, and rule-based fixes (Ferragut). Products use fixed limits or MPC.",
             "<b>Gap:</b> nobody (to our search) studied a <i>learned</i> policy that can be <i>gamed by misreporting</i>, on <i>real</i> sessions with a <i>shared hard grid limit</i>."]),
          P("Data", H2),
          B(["Real ACN-Data: Caltech 31,424 and JPL 33,638 sessions. One day = 96 steps of 15 min. Port max 6.656 kW. Time-of-use prices (peak $0.267/kWh, 12-18h).",
             "Train Jan-Aug 2019, validate Sep-Oct 2019, test Nov 2019-Feb 2020 (85 days). JPL and post-COVID used as transfer tests.",
             "Grid cap = 20% of the typical peak (34.6 kW), so the station is overloaded part of the time."]),
          P("The RL formulation", H2),
          T([["Agent / episode", "Station controller; one episode = one weekday (96 steps)"], ["State", "Per port: occupied, energy needed, time to REPORTED departure, laxity; plus price, time of day, last total power. True departure hidden."],
             ["Action", "Per port, a fraction (0 to 1) of its max useful power"], ["Reward", "Minus electricity cost of the step"],
             ["Costs (budgets)", "Unmet energy at departure (2% budget); battery-wear proxy (share of ports above 80% power)"],
             ["Safety layer", "Subtract the same amount from every request until total power fits the grid limit (hard guarantee)"],
             ["Algorithm", "PPO-Lagrangian + Deep Sets network (works for any number of cars)"]], [32, 142], head=False, zebra=False),
          P("Reports tested: truthful (M0), noisy (M2), late (M3), strategic early reports (M4), plus real recorded reports (M1r, extra).", SMALL)]
    s += [IMG(ROOT / "docs" / "architecture_diagram.jpg", 174),
          P("Flow: real sessions, report model, environment, RL agent, safety projection, back to the environment.", SMALL),
          P("Compared against", H2),
          B(["FCFS, EDF (earliest reported departure), LLF (least spare time), MPC (re-plan each step), and the <b>offline optimum</b> (best possible schedule knowing true departures)."]),
          P("Results so far (85 test days, 5 seeds, no grid violations anywhere)", H2),
          T(e1_rows(), [32, 24, 28, 26, 36, 28]), Spacer(1, 2 * mm),
          B(["<b>Truthful reports:</b> the learned policy ties EDF; it does not beat the rules. It charges gently (wear proxy about 40% lower).",
             "<b>Wrong reports:</b> learned policy about 2.6x worse, rules about 1.2-1.6x, MPC about 30x.",
             "<b>Cheating works:</b> reporting earlier gets the learned policy's driver extra energy in 93-97% of cases; rules only when crowded. Gain grows with how early and with a tighter cap.",
             "<b>Fix:</b> serving a car only until its declared departure removes the gain for every method, but hurts honest drivers whose reports are early by mistake.",
             "<b>Transfer:</b> fine after COVID at Caltech; fails at JPL (28.6% unmet)."]),
          IMG(FIG / "e3.png", 165),
          P("Novel parts", H2),
          B(["Reported departure as something a driver may fake, tested on a learned policy with real data.", "Paired-replay measure of misreporting gain.", "Hard projection + budgets for a shared station.",
             "Comparison with hindsight optimum; test of the Ferragut fix on RL.", "Extra: real driver reports are common and 2-3 h wrong (the '0.2%' in the submitted text is wrong)."]),
          P("Status", H2),
          T([["Done", "Data, simulator, safety layer, baselines, offline optimum, MPC, PPO-Lagrangian (5 seeds), experiments E1, E2, E3, E6, first part of E4, 108 tests passing, code on GitHub"],
             ["In progress", "E4: training with random reports; 6 of 9 runs done, 3 paused (about 1-2 h to finish)"],
             ["Left", "E5 ablation, unprojected RL baseline, SAC-Lagrangian (optional), final report and slides"]], [26, 148], head=False, zebra=False),
          P("Honest points: learned policy needs a warm start copied from LLF (not in design); several constants were our choices (all logged in docs/PROJECT_LOG.md); misreporting is simulated.", SMALL),
          P("Map to mid-review slides", H2),
          T([["1 Title", "cover"], ["2 Problem", "'The problem', 'Why RL'"], ["3 Existing work", "'What exists and our gap'"], ["4 MDP + algorithm + data", "'Data', 'The RL formulation'"],
             ["5 System diagram", "the figure on page 2"], ["6 Novel parts", "'Novel parts'"], ["7 Demo + next plan", "'Results', 'Status'"]], [50, 124], head=False)]
    build(OUT / "SafeCharge_Quick_Guide.pdf", s, "Quick guide")


if __name__ == "__main__":
    fig_e1(); fig_e2(); fig_e3(); fig_train()
    full(); quick()
    print("wrote", OUT)
