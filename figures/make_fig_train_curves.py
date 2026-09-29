#!/usr/bin/env python3
"""Fig: training trajectory of the derivative-supervised chain (test fold).

The derivative-supervised curve is one 160-epoch training path of three
stages at increasing derivative weight, each continued from the previous
stage's final checkpoint (stage boundaries at epochs 80 and 120):

  lambda=0.1  epochs   1- 80   derivative supervision switched on
  lambda=0.3  epochs  81-120   continued from the lambda=0.1 endpoint
  lambda=0.5  epochs 121-160   continued from the lambda=0.3 endpoint

The lambda=0.5 endpoint is the main-text operating point. Validation is
scored every fifth epoch. The H-only curve is a separate 50-epoch
fine-tune on static matrices only, from the same frozen checkpoint; its
own epoch axis restarts at 1.

Reference levels in panel (b) are test-fold medians: frozen 0.708 and
H-only 0.610. All values are relative Frobenius errors.

Data: fig_bulk_train_curves_data.csv, extracted from the representative
seed (20260715) training histories in the shared record. Colours follow
the seaborn "muted" palette used by the other figures.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RED, GREEN, BLUE, GREY = "#c44e52", "#55a868", "#4c72b0", "#888888"
FROZEN_TEST, HONLY_TEST = 0.708, 0.610

rows = list(csv.DictReader(open(Path(__file__).with_name("fig_bulk_train_curves_data.csv"))))
ep = [int(r["epoch"]) for r in rows]
lam = [r["lam"] for r in rows]
jt_h = [float(r["joint_train_h"]) for r in rows]
jt_d = [float(r["joint_train_d"]) for r in rows]
vep = [int(r["epoch"]) for r in rows if r["joint_val_d"]]
jv_h = [float(r["joint_val_h"]) for r in rows if r["joint_val_d"]]
jv_d = [float(r["joint_val_d"]) for r in rows if r["joint_val_d"]]
hep = [int(r["epoch"]) for r in rows if r["honly_train_h"]]
hh = [float(r["honly_train_h"]) for r in rows if r["honly_train_h"]]

# stage bands: (first epoch, last epoch, lambda)
BANDS = [(1, 80, "0.1"), (81, 120, "0.3"), (121, 160, "0.5")]


def band_marks(ax):
    """Vertical stage boundaries and the lambda label of each band."""
    trans = ax.get_xaxis_transform()          # x in data, y in axes fraction
    for first, last, lab in BANDS:
        if first > 1:
            ax.axvline(first - 0.5, color=GREY, lw=0.7, ls=":", alpha=0.7)
        ax.text((first + last) / 2, 0.97, rf"$\lambda={lab}$", transform=trans,
                ha="center", va="top", fontsize=7, color="#666")


fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))

# panel (a): static Hamiltonian along the chain
ax = axes[0]
ax.plot(ep, jt_h, "-", color=RED, lw=1.6, label="joint train")
ax.plot(vep, jv_h, "o", color=RED, ms=4.0, label="joint val")
ax.plot(hep, hh, "--", color=GREEN, lw=1.4, label="H-only train")
ax.set_yscale("log")
ax.set_ylim(2e-4, 8e-3)
ax.set_xlim(0, 160)
ax.set_xlabel("epoch")
ax.set_ylabel(r"rel. Frobenius $H$")
ax.set_title(r"(a) Static $H$", fontsize=9)
ax.legend(loc="lower left", fontsize=7, frameon=False)
band_marks(ax)

# panel (b): matrix response along the chain
ax = axes[1]
ax.plot(ep, jt_d, "-", color=RED, lw=1.6, label="joint train $dH$")
ax.plot(vep, jv_d, "o", color=RED, ms=4.0, label="joint val $dH$")
ax.axhline(FROZEN_TEST, color=BLUE, lw=1.2, ls=":", label="frozen test")
ax.axhline(HONLY_TEST, color=GREEN, lw=1.2, ls="--", label="H-only test")
ax.set_ylim(0.03, 0.80)
ax.set_xlim(0, 160)
ax.set_xlabel("epoch")
ax.set_ylabel(r"mean rel. Frobenius $dH/dR$")
ax.set_title(r"(b) Matrix $dH/dR$", fontsize=9)
ax.legend(loc="center right", fontsize=7, frameon=False)
band_marks(ax)

fig.tight_layout()
fig.savefig("fig_bulk_train_curves.pdf", bbox_inches="tight")
print("wrote fig_bulk_train_curves.pdf")
