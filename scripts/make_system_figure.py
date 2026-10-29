"""Slide-ready system design figure with the full RL loop (state, action, safe power, reward and costs, evaluation).
Run: python scripts/make_system_figure.py  ->  docs/briefing/fig/system_design.png   (needs matplotlib)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(figsize=(13.33, 6.2))
ax.set_xlim(0, 13.33); ax.set_ylim(0, 6.2); ax.axis("off")
NAVY, INK = "#0B2545", "#1B1F2A"

def box(x, y, w, h, title, body, fc, ec, num):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=2))
    ax.add_patch(plt.Circle((x + 0.3, y + h - 0.3), 0.19, fc=ec, ec=ec))
    ax.text(x + 0.3, y + h - 0.3, str(num), ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    ax.text(x + 0.62, y + h - 0.3, title, ha="left", va="center", fontsize=13, fontweight="bold", color=NAVY)
    ax.text(x + 0.2, y + h - 0.62, body, ha="left", va="top", fontsize=11.5, color=INK, linespacing=1.35)

def arrow(p, q, label, lx, ly, color=NAVY, rad=0.0, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=22, lw=2.2, color=color, connectionstyle=f"arc3,rad={rad}"))
    ax.text(lx, ly, label, ha=ha, va="center", fontsize=11.5, color=color, fontweight="bold",
            bbox=dict(fc="white", ec="none", pad=1.5))

# top row: data -> report model -> environment -> evaluation
box(0.2, 3.55, 2.9, 2.2, "ACN-Data", "Real Caltech / JPL\ncharging sessions:\narrival, true departure,\nenergy needed", "#DCE9FA", "#4A7FC1", 1)
box(3.7, 3.55, 2.9, 2.2, "Report model", "Makes the departure the\npolicy sees:\ntruthful, noisy, late,\nor faked early", "#DDEFD9", "#5E9E4E", 2)
box(7.2, 3.55, 2.9, 2.2, "Environment", "Ports, price, grid cap.\nCars leave at their TRUE\ntime; unmet energy is\ncounted then", "#FFF1C9", "#C9A227", 3)
box(10.7, 3.55, 2.5, 2.2, "Evaluation", "Cost, unmet energy,\ngrid violations,\ncheating gain vs\nbaselines, optimum", "#EDEDED", "#777777", 6)
# bottom row: projection <- agent
box(3.6, 0.35, 3.3, 2.1, "Safety projection", "Cuts the requested\npower to the grid cap:\na hard guarantee", "#E7DDF0", "#8B63A8", 5)
box(9.0, 0.35, 3.4, 2.1, "RL agent", "PPO-Lagrangian,\nDeep Sets network.\nMultipliers rise when a\ncost budget is broken", "#F8D5D3", "#C0504D", 4)

arrow((3.1, 4.65), (3.7, 4.65), "", 0, 0)
arrow((6.6, 4.65), (7.2, 4.65), "", 0, 0)
arrow((10.1, 4.65), (10.7, 4.65), "", 0, 0)
ax.text(5.15, 6.0, "reported departure only", ha="center", fontsize=11, style="italic", color="#555555")
arrow((9.4, 3.55), (10.3, 2.45), "state s, reward R,\ncosts c1, c2", 11.2, 3.0, rad=-0.15)
arrow((9.0, 1.4), (6.9, 1.4), "action u\n(0 to 1 per port)", 7.95, 1.95, ha="center")
arrow((5.6, 2.45), (7.9, 3.55), "safe power p\n(total within cap)", 5.0, 3.05, rad=0.1, ha="center")
ax.text(6.67, 0.08, "Loop repeats every 15 minutes, 96 steps per day. The true departure is never shown to the agent.", ha="center", fontsize=11.5, style="italic", color="#555555")
fig.savefig("docs/briefing/fig/system_design.png", dpi=200, bbox_inches="tight", facecolor="white")
