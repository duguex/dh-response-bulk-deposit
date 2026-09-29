"""Uniform-diagonal (potential-zero) artifact audit for FD dH/dR label caches.

Motivation: G1b joint derivative training on the defect-mixed library stalled at
train relfro ~0.85 while the bulk perfect-cell campaign reached 0.29 with the
same label protocol. Leading hypothesis: in charged supercells the plus/minus
SCF runs may settle at slightly different potential zeros, and the central
difference dH = (H+ - H-)/(2 delta) then carries a spurious uniform diagonal
component alpha*I whose Frobenius mass is |alpha|*sqrt(N) -- amplified by
1/(2 delta) = 500 /A per meV of zero mismatch.

Per sample (packed layout: first n_atoms rows are on-site 19x19 blocks, then
n_edges edge blocks):
  fro        Frobenius norm of dh_dr over the active mask
  alpha_med  median over atoms of the per-atom mean on-site diagonal of dh_dr
             (robust estimate of a global constant shift; the physical response
             is local to the displaced atom and only perturbs few atoms)
  chan_frac  |alpha_med| * sqrt(N) / fro  -- artifact-channel mass fraction
  diag_frac  sqrt(sum(diag(on-site blocks)^2)) / fro
  mu_std     std across atoms of per-atom mean diagonal (physical heterogeneity)

Usage:
  python scripts/diff_hamgnn/audit_dh_labels.py \
      --root /mnt/shared/work/dh_response_labels_v2/samples \
      --out docs/analysis/dh_response/label_audit/defect_lib.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np


def charge_of(structure_id: str) -> int:
    m = re.search(r"chg(-?\d+)", structure_id)
    return int(m.group(1)) if m else None


def audit_sample(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    dh = z["dh_dr"].astype(np.float64)
    am = z["active_mask"]
    n_edges = z["edge_index"].shape[1]
    n_rows = dh.shape[0]
    if dh.shape[1] != 361 or n_rows <= n_edges:
        # legacy dense-format rows (QC-failed, excluded from training): record and skip
        n_atoms = -1
        return {
            "sample_id": str(z["sample_id"]), "structure_id": str(z["structure_id"]),
            "q": charge_of(str(z["structure_id"])), "atom": int(z["atom"]), "axis": int(z["axis"]),
            "n_atoms": n_atoms, "n_edges": n_edges, "fro": float("nan"), "alpha_med": float("nan"),
            "alpha_mean": float("nan"), "chan_frac": float("nan"), "diag_frac": float("nan"),
            "mu_std": float("nan"), "mu_maxabs": float("nan"),
            "qc_passed": bool(z["qc_passed"]), "zero_cut_missing_plus": -1, "zero_cut_missing_minus": -1,
        }
    n_atoms = n_rows - n_edges  # 62..66 across vacancy/interstitial stems
    nao = 19

    fro = math.sqrt(float((dh[am] ** 2).sum()))

    onsite = dh[:n_atoms].reshape(n_atoms, nao, nao)
    am_onsite = am[:n_atoms].reshape(n_atoms, nao, nao)
    diag = np.einsum("aii->ai", onsite)
    diag_ok = np.einsum("aii->ai", am_onsite)
    if not diag_ok.all():
        diag = np.where(diag_ok, diag, 0.0)

    mu_atom = diag.mean(axis=1)  # per-atom mean on-site diagonal derivative
    n_total = n_atoms * nao
    alpha_med = float(np.median(mu_atom))
    alpha_mean = float(mu_atom.mean())
    chan = abs(alpha_med) * math.sqrt(n_total)
    diag_mass = math.sqrt(float((diag**2).sum()))

    return {
        "sample_id": str(z["sample_id"]),
        "structure_id": str(z["structure_id"]),
        "q": charge_of(str(z["structure_id"])),
        "atom": int(z["atom"]),
        "axis": int(z["axis"]),
        "n_atoms": n_atoms,
        "n_edges": n_edges,
        "fro": fro,
        "alpha_med": alpha_med,
        "alpha_mean": alpha_mean,
        "chan_frac": chan / fro if fro > 0 else float("nan"),
        "diag_frac": diag_mass / fro if fro > 0 else float("nan"),
        "mu_std": float(mu_atom.std()),
        "mu_maxabs": float(np.abs(mu_atom).max()),
        "qc_passed": bool(z["qc_passed"]),
        "zero_cut_missing_plus": int(z["zero_cutoff_missing_keys_plus"].shape[0]),
        "zero_cut_missing_minus": int(z["zero_cutoff_missing_keys_minus"].shape[0]),
    }


def summarize(rows: list[dict], label: str) -> str:
    rows = [r for r in rows if not math.isnan(r.get("fro", float("nan")))]
    by_group: dict[str, list[dict]] = {}
    for r in rows:
        q = r["q"]
        g = "q=0" if q == 0 else ("|q|=3" if abs(q) == 3 else ("|q|=6" if abs(q) == 6 else f"q={q}"))
        by_group.setdefault(g, []).append(r)
    lines = [f"## Group summary -- {label}", "",
             "| group | n | med \\|alpha_med\\| | med chan_frac | p90 chan_frac | med diag_frac | med mu_std |",
             "|---|---|---|---|---|---|---|"]
    for g in sorted(by_group, key=lambda x: (x != "q=0", x)):
        rs = by_group[g]
        med = lambda k: float(np.nanmedian([abs(r[k]) for r in rs]))
        p90 = lambda k: float(np.nanpercentile([r[k] for r in rs], 90))
        lines.append(
            f"| {g} | {len(rs)} | {med('alpha_med'):.4g} | {med('chan_frac'):.4g} "
            f"| {p90('chan_frac'):.4g} | {med('diag_frac'):.4g} | {med('mu_std'):.4g} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dataset-label", default="")
    args = ap.parse_args()

    files = sorted(args.root.glob("*.npz"))
    rows = [audit_sample(p) for p in files]
    n_dense = sum(1 for r in rows if math.isnan(r["fro"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    label = args.dataset_label or args.root.name
    print(f"audited {len(rows)} samples from {args.root} ({n_dense} legacy-dense skipped, "
          f"{len(rows) - n_dense} packed)")
    print(summarize(rows, label))

    worst = sorted((r for r in rows if not math.isnan(r["chan_frac"])), key=lambda r: -r["chan_frac"])[:10]
    print("\nTop-10 chan_frac outliers:")
    for r in worst:
        print(f"  {r['sample_id']:36s} q={r['q']!s:>3} chan_frac={r['chan_frac']:.4g} alpha_med={r['alpha_med']:.4g}")

    # paired within-stem comparison: charged vs q=0 of the same stem
    stems: dict[str, dict[int, list[float]]] = {}
    for r in rows:
        stem = re.sub(r"chg-?\d+$", "", r["structure_id"])
        stems.setdefault(stem, {}).setdefault(r["q"], []).append(r["chan_frac"])
    ratios = []
    for stem, qs in stems.items():
        if 0 in qs and any(q != 0 for q in qs):
            base = float(np.median(qs[0]))
            for q, vals in qs.items():
                if q != 0:
                    ratios.append((stem, q, float(np.median(vals)) / base if base > 0 else float("nan")))
    if ratios:
        rr = [x[2] for x in ratios if not math.isnan(x[2])]
        print(f"\nPaired charged/q0 chan_frac ratio: n={len(rr)} median={np.median(rr):.3g} "
              f"p25={np.percentile(rr,25):.3g} p75={np.percentile(rr,75):.3g}")


if __name__ == "__main__":
    main()
