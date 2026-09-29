#!/usr/bin/env python3
"""Phase 0 Track B: H + dH/dR diagnostics on Perf_chg0 only.

Does not use the 540-campaign evaluator contract. Loads the 12 FD label
files for structure_id=Perf_chg0 from an existing cache and scores one or
more checkpoints.

Example:
  python scripts/diff_hamgnn/bulk_perf0_diag.py \\
    --cache-root /mnt/shared/work/dh_response_labels_v2 \\
    --graph data/npz/Perf_chg0.npz \\
    --ckpt-tag g1a=/mnt/shared/work/dh_response_g1a/g1a_embq_fix_r2.ckpt \\
    --out-dir /mnt/shared/work/dh_response_bulk_diag/perf0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DELTA = 0.001
HARTREE_TO_MEV = 27211.386245988


def _relfro(pred: np.ndarray, target: np.ndarray) -> float:
    denom = float(np.linalg.norm(target))
    if denom <= 0.0:
        return float("nan")
    return float(np.linalg.norm(pred - target) / denom)


def _cosine(pred: np.ndarray, target: np.ndarray) -> float:
    a = float(np.linalg.norm(pred))
    b = float(np.linalg.norm(target))
    if a <= 0.0 or b <= 0.0:
        return float("nan")
    return float(np.dot(pred, target) / (a * b))


def _parse_ckpt_tags(items: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--ckpt-tag must be name=path, got {item!r}")
        name, path = item.split("=", 1)
        out.append((name.strip(), Path(path).expanduser().resolve()))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--structure-id", default="Perf_chg0")
    p.add_argument("--ckpt-tag", action="append", default=[], dest="ckpt_tags")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--hamgnn-root", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--delta", type=float, default=DELTA)
    args = p.parse_args(argv)

    if not args.ckpt_tags:
        raise SystemExit("pass at least one --ckpt-tag name=path")

    from scripts.diff_hamgnn import DifferentiableHamGNN
    from scripts.diff_hamgnn.defaults import resolve_paths
    from scripts.diff_hamgnn.evaluate_dh_response import model_component_fd

    cache_root = args.cache_root.expanduser().resolve()
    samples = sorted((cache_root / "samples").glob(f"{args.structure_id}_atom*_axis*.npz"))
    if len(samples) != 12:
        print(f"ERROR: expected 12 samples for {args.structure_id}, found {len(samples)}", file=sys.stderr)
        return 2

    records = []
    for f in samples:
        z = np.load(f)
        atom = int(z["atom"]) if "atom" in z.files else int(f.stem.split("atom")[1].split("_")[0])
        axis = int(z["axis"]) if "axis" in z.files else int(f.stem.split("axis")[1])
        records.append(
            {
                "path": f,
                "atom": atom,
                "axis": axis,
                "h_mid": np.asarray(z["h_mid"], dtype=np.float64),
                "dh_dr": np.asarray(z["dh_dr"], dtype=np.float64),
                "mask": np.asarray(z["active_mask"], dtype=bool),
            }
        )
    records.sort(key=lambda r: (r["atom"], r["axis"]))

    graph = args.graph.expanduser().resolve()
    if not graph.is_file():
        # try repo-relative
        alt = (_REPO / args.graph).resolve()
        if alt.is_file():
            graph = alt
        else:
            print(f"ERROR: graph not found: {args.graph}", file=sys.stderr)
            return 2

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "structure_id": args.structure_id,
        "graph": str(graph),
        "cache_root": str(cache_root),
        "n_components": len(records),
        "delta_angstrom": float(args.delta),
        "arms": {},
    }

    for tag, ckpt in _parse_ckpt_tags(args.ckpt_tags):
        if not ckpt.is_file():
            print(f"ERROR: missing ckpt {ckpt}", file=sys.stderr)
            return 2
        paths = resolve_paths(args.hamgnn_root, args.config, None, ckpt)
        missing = paths.missing()
        if missing:
            for m in missing:
                print(f"ERROR: missing {m}", file=sys.stderr)
            return 2

        m = DifferentiableHamGNN.from_files(
            paths.config,
            paths.ckpt,
            paths.hamgnn_root,
            device=args.device,
            freeze_weights=True,
            emb_q_slots=24,
        )
        m.set_graph(graph)

        with torch.no_grad():
            H = m.forward_H(pos=None).detach().cpu().numpy().astype(np.float64)
        mask0 = records[0]["mask"]
        h_active_pred = H[mask0]
        h_active_ref = records[0]["h_mid"][mask0]
        h_rel = _relfro(h_active_pred, h_active_ref)
        h_mae_mev = float(np.mean(np.abs(h_active_pred - h_active_ref)) * HARTREE_TO_MEV)

        rows = []
        d_rels = []
        d_cos = []
        d_ratios = []
        for rec in records:
            d_pred = (
                model_component_fd(m, rec["atom"], rec["axis"], float(args.delta))
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            mask = rec["mask"]
            pred = d_pred[mask]
            ref = rec["dh_dr"][mask]
            rel = _relfro(pred, ref)
            cos = _cosine(pred, ref)
            ratio = float(np.linalg.norm(pred) / np.linalg.norm(ref)) if np.linalg.norm(ref) > 0 else float("nan")
            d_rels.append(rel)
            d_cos.append(cos)
            d_ratios.append(ratio)
            rows.append(
                {
                    "atom": rec["atom"],
                    "axis": rec["axis"],
                    "relative_frobenius_error": rel,
                    "frobenius_cosine": cos,
                    "norm_ratio": ratio,
                    "dh_mae_mev_per_angstrom": float(np.mean(np.abs(pred - ref)) * HARTREE_TO_MEV),
                }
            )

        arm = {
            "ckpt": str(ckpt),
            "h_relfro": h_rel,
            "h_mae_mev": h_mae_mev,
            "dh_relfro_median": float(np.median(d_rels)),
            "dh_relfro_mean": float(np.mean(d_rels)),
            "dh_cosine_median": float(np.median(d_cos)),
            "dh_norm_ratio_median": float(np.median(d_ratios)),
            "components": rows,
        }
        summary["arms"][tag] = arm
        print(
            f"[{tag}] H_rel={h_rel:.4f} H_mae_meV={h_mae_mev:.3f} "
            f"dH_rel_med={arm['dh_relfro_median']:.4f} dH_cos_med={arm['dh_cosine_median']:.4f}"
        )

    out_json = out_dir / f"{args.structure_id}_diag.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
