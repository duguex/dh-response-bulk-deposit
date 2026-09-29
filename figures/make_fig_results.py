#!/usr/bin/env python3
"""Fig. 1 (merged): the single results figure of the submission.

Panels, in the text's first-citation order:
  (a) static H training trajectory                [was fig:curves panel (a)]
  (b) dH/dR training trajectory                   [was fig:curves panel (b)]
  (c) derivative-weight sweep, dH/dR vs lambda    [was fig:ladder panel (a)]
  (d) static-versus-derivative trade-off          [was fig:ladder panel (b)]

The former third figure of this script --- the error-budget bar panel, which
was fig:budget --- became a table (tab:budget) when the bars figure of the
main text (fig:bars) was dropped: both were four-number bar charts whose
numbers already stand in the prose. fig:bars' fourth arm, the 40-epoch
lambda=0.1 reference run, is carried by panels (c) and (d) below, labelled
"0.1 (40 ep)"; it is a reference arm, not a rung of the canonical chain.

Data provenance (unchanged from the former single-panel figures):
  (a,b) fig_bulk_train_curves_data.csv --- representative seed 20260715, the
        160-epoch three-stage chain (lambda=0.1/0.3/0.5 at epochs
        1-80/81-120/121-160), validation every fifth epoch; H-only is a
        separate 50-epoch static fine-tune from the same baseline checkpoint
        with its own epoch axis. Reference levels in (b) are test-fold
        component means: baseline 0.708, H-only 0.610.
  (c,d) canonical sweep, component-mean relative Frobenius errors on the
        held-out test fold; marker fill encodes seed count (3/2/1).

The source width equals the submission text width (370.6 pt = 5.15 in), so
the point sizes below are the point sizes in the final PDF.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

RED, GREEN, BLUE, GREY = "#c44e52", "#55a868", "#1f77b4", "#888888"
FROZEN_TEST, HONLY_TEST = 0.708, 0.610

# ------------------------------------------------------------- (a,b) data
rows = list(csv.DictReader(open(Path(__file__).with_name("fig_bulk_train_curves_data.csv"))))
ep = [int(r["epoch"]) for r in rows]
jt_h = [float(r["joint_train_h"]) for r in rows]
jt_d = [float(r["joint_train_d"]) for r in rows]
vep = [int(r["epoch"]) for r in rows if r["joint_val_d"]]
jv_h = [float(r["joint_val_h"]) for r in rows if r["joint_val_d"]]
jv_d = [float(r["joint_val_d"]) for r in rows if r["joint_val_d"]]
hep = [int(r["epoch"]) for r in rows if r["honly_train_h"]]
hh = [float(r["honly_train_h"]) for r in rows if r["honly_train_h"]]

BANDS = [(1, 80, "0.1"), (81, 120, "0.3"), (121, 160, "0.5")]

# ------------------------------------------------------------ (c,d) data
LADDER = [
    # (lambda, dH relfro, H relfro, seeds)
    (0.03, 0.240, 0.0008, 1),
    (0.1, 0.290, 0.0010, 3),    # 40-epoch reference arm (was fig:bars)
    (0.1, 0.175, 0.0007, 3),    # canonical chain, 80 epochs
    (0.3, 0.118, 0.0014, 3),
    (0.5, 0.095, 0.0019, 3),    # main-text operating point
    (1.0, 0.061, 0.0045, 2),
    (2.0, 0.0715, 0.0067, 1),   # probe from the deepest checkpoint
]
FROZEN = (0.708, 0.00435)
HONLY = (0.610, 0.00042)

# The 40-epoch lambda=0.1 run is a reference arm, not a rung of the
# canonical chain: connecting it would kink the frontier line backwards in x.
REF_ARM = (0.1, 0.290)
CHAIN = [p for p in LADDER if (p[0], p[1]) != REF_ARM]


def _marker(seeds):
    if seeds == 3:
        return {"ms": 4.0, "mfc": BLUE}
    if seeds == 2:
        return {"ms": 3.7, "mfc": "#9ecae1"}
    return {"ms": 3.2, "mfc": "white"}


def band_marks(ax, y=0.97):
    """Vertical stage boundaries and the lambda label of each band."""
    trans = ax.get_xaxis_transform()          # x in data, y in axes fraction
    va = "top" if y > 0.9 else "center"
    for first, last, lab in BANDS:
        if first > 1:
            ax.axvline(first - 0.5, color=GREY, lw=0.6, ls=":", alpha=0.7)
        ax.text((first + last) / 2, y, rf"$\lambda={lab}$", transform=trans,
                ha="center", va=va, fontsize=6, color="#666")


def panel_label(ax, letter):
    # just above the top-left corner of the axes box; the lambda labels of
    # (a,b) sit inside the box and the reference-line labels at the right
    ax.text(0.0, 1.03, f"({letter})", transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="bottom", ha="left")


fig = plt.figure(figsize=(5.15, 4.15))
gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.32)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0])
ax_d = fig.add_subplot(gs[1, 1])
fig.subplots_adjust(left=0.135, right=0.985, top=0.925, bottom=0.095)

# (a) static Hamiltonian along the chain; series are labelled in place
ax_a.plot(ep, jt_h, "-", color=RED, lw=1.3)
ax_a.plot(vep, jv_h, "o", color=RED, ms=2.6)
ax_a.plot(hep, hh, "--", color=GREEN, lw=1.1)
ax_a.set_yscale("log")
ax_a.set_ylim(1.1e-4, 8e-3)
ax_a.set_xlim(0, 160)
ax_a.set_xlabel("epoch", fontsize=7.5)
ax_a.set_ylabel(r"rel. Frobenius $H$", fontsize=7.5)
ax_a.tick_params(labelsize=6.5)
band_marks(ax_a)
# the red pair is labelled once, in the empty band above the chain; the
# green fine-tune ends at epoch 50 and is labelled in the band below it
ax_a.annotate("joint train / val", (10, 3.4e-3), xytext=(0, 0),
              textcoords="offset points", fontsize=6, color=RED,
              ha="left", va="bottom")
ax_a.annotate("H-only train", (52, 1.75e-4), xytext=(0, 0),
              textcoords="offset points", fontsize=6, color=GREEN,
              ha="left", va="top")

# (b) matrix response along the chain; reference levels labelled in place
ax_b.plot(ep, jt_d, "-", color=RED, lw=1.3)
ax_b.plot(vep, jv_d, "o", color=RED, ms=2.6)
ax_b.axhline(FROZEN_TEST, color=GREY, lw=1.0, ls=":")
ax_b.axhline(HONLY_TEST, color=GREEN, lw=1.0, ls="--")
ax_b.set_ylim(0.03, 0.80)
ax_b.set_xlim(0, 160)
ax_b.set_xlabel("epoch", fontsize=7.5)
ax_b.set_ylabel(r"mean rel. Frobenius $dH/dR$", fontsize=7.5)
ax_b.tick_params(labelsize=6.5)
# (b)'s curve ends low, so its stage labels go to the central blank band
band_marks(ax_b, y=0.55)
# both reference labels sit just below their lines: the region between the
# chain (below ~0.2) and the H-only level is empty, and the lambda=0.5 stage
# label sits at the top right of this panel
ax_b.annotate("baseline test", (158, FROZEN_TEST), xytext=(0, -2),
              textcoords="offset points", fontsize=6, color=GREY,
              ha="right", va="top")
ax_b.annotate("H-only test", (158, HONLY_TEST), xytext=(0, -2),
              textcoords="offset points", fontsize=6, color=GREEN,
              ha="right", va="top")
ax_b.annotate("joint train / val $dH$", (86, jt_d[85]), xytext=(0, 6),
              textcoords="offset points", fontsize=6, color=RED,
              ha="left", va="bottom")

# (c) dose-response in derivative weight
ax_c.axhline(FROZEN[0], color=GREY, lw=1.0, ls=":")
ax_c.axhline(HONLY[0], color="#aaa", lw=1.0, ls="--")
for (l, d, h, seeds) in LADDER:
    ax_c.plot(l, d, "o", mec=BLUE, mew=1.0, **(_marker(seeds)))
ax_c.set_xscale("log")
ax_c.set_xticks([0.03, 0.1, 0.3, 1.0, 2.0])
ax_c.set_xticklabels(["0.03", "0.1", "0.3", "1", "2"])
ax_c.set_xlabel(r"derivative weight $\lambda$", fontsize=7.5)
ax_c.set_ylabel(r"test $dH/dR$ relfro", fontsize=7.5)
ax_c.set_ylim(0, 0.98)
ax_c.set_xlim(0.025, 3.0)
ax_c.tick_params(labelsize=6.5)
ax_c.annotate("baseline", (2.85, FROZEN[0]), xytext=(0, 3),
              textcoords="offset points", fontsize=6, color=GREY,
              ha="right", va="bottom")
ax_c.annotate("H-only", (2.85, HONLY[0]), xytext=(0, 3),
              textcoords="offset points", fontsize=6, color="#aaa",
              ha="right", va="bottom")
# the two lambda=0.1 runs sit at the same weight and must be told apart
ax_c.annotate("0.1 (40 ep)", (0.1, 0.290), xytext=(0, 6),
              textcoords="offset points", fontsize=6, color=BLUE,
              ha="center", va="bottom")
ax_c.annotate("0.1 (80 ep)", (0.1, 0.175), xytext=(-5, -4),
              textcoords="offset points", fontsize=6, color=BLUE,
              ha="right", va="top")

# (d) value-slope trade-off frontier
# the two baseline squares sit near the right edge: their labels go to the
# left of the markers so that they do not run into the panel spine
for (dh, h, lab, ls, c) in [(FROZEN[0], FROZEN[1], "baseline", ":", GREY),
                            (HONLY[0], HONLY[1], "H-only", "--", "#aaa")]:
    ax_d.plot(dh, h, "s", ms=3.4, mec=c, mfc="white", mew=1.0)
    ax_d.annotate(lab, (dh, h), xytext=(-5, 2), textcoords="offset points",
                  fontsize=6, color=c, ha="right")
dhs = [d for _, d, _, _ in CHAIN]
hs = [h for _, _, h, _ in CHAIN]
ax_d.plot(dhs, hs, "-", color=BLUE, lw=1.0, alpha=0.9)
LABELS = {
    (0.03, 0.240): ("0.03", (0, -8), "center", "baseline"),
    (0.1, 0.290): ("0.1 (40 ep)", (5, -1), "left", "baseline"),
    (0.1, 0.175): ("0.1 (80 ep)", (0, -8), "center", "baseline"),
    (0.3, 0.118): ("0.3", (4, 3), "left", "baseline"),
    (0.5, 0.095): ("0.5", (4, 3), "left", "baseline"),
    (1.0, 0.061): ("1.0", (-5, -3), "right", "top"),
    (2.0, 0.0715): ("2.0", (-5, -1), "right", "baseline"),
}
for (l, d, h, seeds) in LADDER:
    ax_d.plot(d, h, "o", mec=BLUE, mew=1.0, **(_marker(seeds)))
    ent = LABELS[(l, d)]
    txt, off, ha = ent[:3]
    va = ent[3] if len(ent) > 3 else "baseline"
    ax_d.annotate(txt, (d, h), xytext=off, textcoords="offset points",
                  fontsize=6, color=BLUE, ha=ha, va=va)
ax_d.axhline(FROZEN[1], color=GREY, lw=0.7, ls=":")
ax_d.set_xscale("log")
ax_d.set_yscale("log")
ax_d.set_xlabel(r"test $dH/dR$ relfro", fontsize=7.5)
ax_d.set_ylabel(r"test $H$ relfro", fontsize=7.5)
ax_d.set_xlim(0.043, 1.0)
ax_d.set_ylim(3e-4, 1e-2)
ax_d.tick_params(labelsize=6.5)
# the dotted line needs no label of its own: the frozen square sits on it

for ax, letter in [(ax_a, "a"), (ax_b, "b"), (ax_c, "c"), (ax_d, "d")]:
    panel_label(ax, letter)

fig.savefig("fig_results.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("fig_results.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
print("wrote fig_results.pdf / fig_results.png (4-panel merged results figure)")
