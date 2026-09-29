#!/usr/bin/env python3
"""Bulk Track B fold eval: H + dH/dR on a bulk label cache (not 540-campaign).

Scores one or more checkpoints on train/val/test folds from dh_bulk_manifest.

Example:
  python scripts/diff_hamgnn/bulk_fold_eval.py \\
    --manifest config/dh_bulk_manifest.yaml \\
    --cache-root /mnt/shared/work/dh_response_bulk_labels_v1 \\
    --ckpt-tag frozen=/mnt/shared/work/dh_response_g1a/g1a_embq_fix_r2.ckpt \\
    --ckpt-tag h_only=/mnt/shared/work/dh_response_bulk_train/h_only_best.ckpt \\
    --ckpt-tag joint=/mnt/shared/work/dh_response_bulk_train/joint_best.ckpt \\
    --out /mnt/shared/work/dh_response_bulk_train/fold_eval.json \\
    --device cuda:0
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


def _relfro(pred: torch.Tensor, target: torch.Tensor) -> float:
    denom = float(torch.linalg.norm(target).item())
    if denom <= 0.0:
        return float("nan")
    return float(torch.linalg.norm(pred - target).item() / denom)


def _parse_tags(items: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--ckpt-tag must be name=path, got {item!r}")
        name, path = item.split("=", 1)
        out.append((name.strip(), Path(path).expanduser().resolve()))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--ckpt-tag", action="append", default=[], dest="ckpt_tags")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--hamgnn-root", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--folds-train", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--folds-val", nargs="+", type=int, default=[3])
    p.add_argument("--folds-test", nargs="+", type=int, default=[4])
    p.add_argument("--delta", type=float, default=None)
    args = p.parse_args(argv)
    if not args.ckpt_tags:
        raise SystemExit("pass at least one --ckpt-tag name=path")

    from scripts.diff_hamgnn import DifferentiableHamGNN
    from scripts.diff_hamgnn.build_dh_label_cache import _load_manifest, _structure_rows
    from scripts.diff_hamgnn.defaults import resolve_paths
    from scripts.diff_hamgnn.evaluate_dh_response import model_component_fd

    tags = _parse_tags(args.ckpt_tags)
    manifest_path = args.manifest.expanduser().resolve()
    manifest, _ = _load_manifest(manifest_path)
    delta = float(args.delta if args.delta is not None else manifest.get("delta_angstrom", 0.001))
    axes = [int(a) for a in manifest["axes"]]
    entries, fails = _structure_rows(manifest, manifest_path, load_supplied_structures=True)
    if fails:
        print("WARN structure failures:", fails, file=sys.stderr)

    fold_sets = {
        "train": set(args.folds_train),
        "val": set(args.folds_val),
        "test": set(args.folds_test),
    }
    by_split: dict[str, list[dict]] = {k: [] for k in fold_sets}
    for e in entries:
        f = int(e["fold"])
        for name, fs in fold_sets.items():
            if f in fs:
                by_split[name].append(e)

    cache_root = args.cache_root.expanduser().resolve()

    def load_labels(e: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
        sid = str(e["structure_id"])
        samples = sorted((cache_root / "samples").glob(f"{sid}_atom*_axis*.npz"))
        atoms = [int(a) for a in e["probe_atoms"].tolist()]
        pairs = [(a, ax) for a in atoms for ax in axes]
        by_key: dict[tuple[int, int], Path] = {}
        for f in samples:
            z = np.load(f)
            atom = int(z["atom"]) if "atom" in z.files else int(f.stem.split("atom")[1].split("_")[0])
            axis = int(z["axis"]) if "axis" in z.files else int(f.stem.split("axis")[1])
            by_key[(atom, axis)] = f
        missing = [k for k in pairs if k not in by_key]
        if missing:
            raise FileNotFoundError(f"{sid} missing labels {missing}")
        ordered = [by_key[k] for k in pairs]
        recs = [np.load(f) for f in ordered]
        h_mid = np.asarray(recs[0]["h_mid"], dtype=np.float32)
        mask = np.asarray(recs[0]["active_mask"], dtype=bool)
        dh = np.stack([np.asarray(r["dh_dr"], dtype=np.float32) for r in recs], axis=0)
        return h_mid, dh, mask, pairs

    labels = {str(e["structure_id"]): load_labels(e) for e in entries}

    summary: dict = {
        "delta_angstrom": delta,
        "n_structures": len(entries),
        "splits": {k: [str(e["structure_id"]) for e in v] for k, v in by_split.items()},
        "arms": {},
    }

    for tag, ckpt in tags:
        paths = resolve_paths(args.hamgnn_root, args.config, None, ckpt)
        m = DifferentiableHamGNN.from_files(
            paths.config,
            paths.ckpt,
            paths.hamgnn_root,
            device=args.device,
            freeze_weights=True,
            emb_q_slots=24,
        )
        arm: dict = {"ckpt": str(ckpt), "splits": {}}
        for split_name, split_entries in by_split.items():
            h_errs: list[float] = []
            d_errs: list[float] = []
            per = []
            for e in split_entries:
                sid = str(e["structure_id"])
                h_mid, dh_dr, mask, pairs = labels[sid]
                m.set_graph(Path(e["path"]).resolve())
                with torch.no_grad():
                    H = m.forward_H(pos=None)
                t = torch.as_tensor(h_mid[mask], device=H.device)
                h_err = _relfro(H[mask], t)
                comp = []
                for i, (atom, axis) in enumerate(pairs):
                    d_pred = model_component_fd(m, atom, int(axis), delta)
                    d_t = torch.as_tensor(dh_dr[i][mask], device=H.device)
                    derr = _relfro(d_pred[mask], d_t)
                    comp.append({"atom": atom, "axis": axis, "dH_relfro": derr})
                    d_errs.append(derr)
                h_errs.append(h_err)
                per.append(
                    {
                        "structure_id": sid,
                        "fold": int(e["fold"]),
                        "H_relfro": h_err,
                        "dH_relfro_mean": float(np.mean([c["dH_relfro"] for c in comp])),
                        "components": comp,
                    }
                )
            arm["splits"][split_name] = {
                "n": len(split_entries),
                "H_relfro_mean": float(np.mean(h_errs)) if h_errs else float("nan"),
                "H_relfro_median": float(np.median(h_errs)) if h_errs else float("nan"),
                "dH_relfro_mean": float(np.mean(d_errs)) if d_errs else float("nan"),
                "dH_relfro_median": float(np.median(d_errs)) if d_errs else float("nan"),
                "structures": per,
            }
            s = arm["splits"][split_name]
            print(
                f"[{tag}] {split_name}: H mean={s['H_relfro_mean']:.4f} med={s['H_relfro_median']:.4f} | "
                f"dH mean={s['dH_relfro_mean']:.4f} med={s['dH_relfro_median']:.4f}",
                flush=True,
            )
        summary["arms"][tag] = arm
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
