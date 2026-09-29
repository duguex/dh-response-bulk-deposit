#!/usr/bin/env python3
"""Fine-tune repaired M4 on static H and optional dH/dR labels.

Default (Track B / G1b): joint loss on each structure

    L = L_H + lambda_d * mean_c L_G_c

with L_H = relfro(H_θ(R0), h_mid) and L_G_c = relfro(G_θ^FD_c, dh_dr_c)
using the same central-difference δ as the OpenMX labels. Gradients are
taken on the single scalar L (mean over components — never unscaled sum).

Track A static path: ``--static-only`` (or ``--lambda-d 0``) trains **H only**,
skips finite-difference derivative forwards, and writes ``g1_static`` metadata.
Starting point: G1a-repaired checkpoint (24-slot charge embedding).

Data split follows the preregistered folds (5 folds x 3 stems): folds
[0,1,2] train, [3] validation, [4] test.

Read-only with respect to the label cache and base ckpt; writes only the
output ckpt paths and a sidecar ``*_history.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DELTA = 0.001  # campaign delta_angstrom (preregistered)


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _relfro(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Deprecated local alias — use joint_loss.relfro."""
    from scripts.diff_hamgnn.joint_loss import relfro

    return relfro(pred, target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="output ckpt stem (adds _best.ckpt / _final.ckpt)")
    parser.add_argument("--ckpt", type=Path, default=None, help="base ckpt (default: G1a-repaired r2)")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--hamgnn-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--lambda-d",
        type=float,
        default=0.1,
        dest="lambda_d",
        help="weight on mean_c L_G (default 0.1; L_G/L_H ~ 1e2 on frozen bulk)",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Track A: force lambda_d=0 and skip dH/dR FD terms",
    )
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--folds-train", nargs="+", type=int, default=[0, 1, 2], dest="folds_train"
    )
    parser.add_argument("--folds-val", nargs="+", type=int, default=[3], dest="folds_val")
    parser.add_argument("--folds-test", nargs="+", type=int, default=[4], dest="folds_test")
    parser.add_argument("--eval-every", type=int, default=10)
    args = parser.parse_args(argv)

    from scripts.diff_hamgnn import DifferentiableHamGNN
    from scripts.diff_hamgnn.build_dh_label_cache import _load_manifest, _structure_rows
    from scripts.diff_hamgnn.defaults import resolve_paths
    from scripts.diff_hamgnn.evaluate_dh_response import model_component_fd
    from scripts.diff_hamgnn.joint_loss import backward_joint, joint_loss, relfro

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.static_only:
        args.lambda_d = 0.0

    paths = resolve_paths(
        args.hamgnn_root,
        args.config,
        None,
        args.ckpt or Path("/mnt/shared/work/dh_response_g1a/g1a_embq_fix_r2.ckpt"),
    )
    missing = paths.missing()
    if missing:
        for p in missing:
            print(f"ERROR: missing {p}", file=sys.stderr)
        return 2
    if not Path(paths.ckpt).is_file():
        print(f"ERROR: base ckpt not found: {paths.ckpt}", file=sys.stderr)
        return 2

    manifest_path = _resolve(args.manifest)
    manifest, _ = _load_manifest(manifest_path)
    delta = float(manifest.get("delta_angstrom", DELTA))
    if abs(delta - DELTA) > 1e-12:
        print(f"WARN: manifest delta {delta} != preregistered {DELTA}", file=sys.stderr)
    entries, structure_failures = _structure_rows(
        manifest, manifest_path, load_supplied_structures=True
    )
    for f in structure_failures:
        print(f"WARN: {f}", file=sys.stderr)

    cache_root = _resolve(args.cache_root)
    folds_train = set(args.folds_train)
    folds_val = set(args.folds_val)
    folds_test = set(args.folds_test)
    if folds_train & folds_val or folds_train & folds_test or folds_val & folds_test:
        print("ERROR: fold sets overlap", file=sys.stderr)
        return 2

    split = {"train": [], "val": [], "test": []}
    for e in entries:
        fold = int(e["fold"])
        if fold in folds_train:
            split["train"].append(e)
        elif fold in folds_val:
            split["val"].append(e)
        elif fold in folds_test:
            split["test"].append(e)
        else:
            print(f"WARN: structure {e['structure_id']} fold {fold} not in any split", file=sys.stderr)
    print(
        f"split: train {len(split['train'])} structures / val {len(split['val'])} / "
        f"test {len(split['test'])}"
    )

    # Load labels (h_mid, dh_dr, active_mask) once per structure.
    def _load_labels(e: dict) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        sid = str(e["structure_id"])
        sample_files = sorted((cache_root / "samples").glob(f"{sid}_*.npz"))
        if not sample_files:
            raise FileNotFoundError(f"no cache samples for {sid}")
        recs = [np.load(f) for f in sample_files]
        h_mid = np.asarray(recs[0]["h_mid"], dtype=np.float32)
        mask = np.asarray(recs[0]["active_mask"], dtype=np.bool_)
        if float(args.lambda_d) == 0.0:
            return h_mid, None, mask
        atoms = [int(a) for a in e["probe_atoms"].tolist()]
        axes = [int(a) for a in manifest["axes"]]
        expect = len(atoms) * len(axes)

        def _key(f: Path) -> tuple[int, int]:
            name = f.stem.split(f"{sid}_")[1]
            atom = int(name.split("_axis")[0].split("atom")[1])
            axis = int(name.split("_axis")[1])
            return (atom, axis)

        by_pair: dict[tuple[int, int], np.ndarray] = {}
        for fpath, rec in zip(sample_files, recs):
            pair = _key(fpath)
            by_pair[pair] = np.asarray(rec["dh_dr"], dtype=np.float32)
        ordered_pairs = [(a, ax) for a in atoms for ax in axes]
        missing = [p for p in ordered_pairs if p not in by_pair]
        if missing:
            raise ValueError(f"{sid}: missing label pairs {missing}")
        extra = len(by_pair) - len(ordered_pairs)
        if extra > 0:
            print(
                f"note: {sid}: ignored {extra} non-probe samples",
                file=sys.stderr,
            )
        if len(ordered_pairs) != expect:
            raise ValueError(f"{sid}: {len(ordered_pairs)} pairs, expected {expect}")
        dh_dr = np.stack([by_pair[p] for p in ordered_pairs], axis=0)
        return h_mid, dh_dr, mask

    labels: dict[str, tuple[np.ndarray, np.ndarray | None, np.ndarray]] = {}
    for e in split["train"] + split["val"] + split["test"]:
        labels[str(e["structure_id"])] = _load_labels(e)

    m = DifferentiableHamGNN.from_files(
        paths.config, paths.ckpt, paths.hamgnn_root,
        device=args.device, freeze_weights=False, emb_q_slots=24,
    )
    print(f"loaded {paths.ckpt} (emb_q {m.model.representation.emb.emb_q.weight.shape})")

    optimizer = torch.optim.Adam(m.model.parameters(), lr=args.lr)
    lambda_d = float(args.lambda_d)

    def _structure_scalars(e: dict, *, train: bool) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Build L_H and per-component L_G list (grad-enabled iff train)."""
        sid = str(e["structure_id"])
        m.set_graph(Path(e["path"]).resolve())
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            H = m.forward_H(pos=None)
            h_mid, dh_dr, mask_np = labels[sid]
            mask = torch.as_tensor(mask_np, device=H.device, dtype=torch.bool)
            t = torch.as_tensor(h_mid, device=H.device, dtype=H.dtype)
            loss_h = relfro(H[mask], t[mask])
            if lambda_d == 0.0 or dh_dr is None:
                return loss_h, None
            atoms = [int(a) for a in e["probe_atoms"].tolist()]
            axes = [int(a) for a in manifest["axes"]]
            dh = torch.as_tensor(dh_dr, device=H.device, dtype=H.dtype)
            comps: list[torch.Tensor] = []
            k = 0
            for atom in atoms:
                for axis in axes:
                    d_pred = model_component_fd(m, atom, int(axis), delta)
                    comps.append(relfro(d_pred[mask], dh[k][mask]))
                    k += 1
            return loss_h, comps

    def _structure_loss(e: dict) -> tuple[float, float]:
        """Eval: (h_rel_err, d_rel_err); d is nan when static-only."""
        loss_h, comps = _structure_scalars(e, train=False)
        _L, h, g = joint_loss(loss_h, comps, lambda_d)
        h_err = float(h.detach())
        if lambda_d == 0.0:
            return h_err, float("nan")
        return h_err, float(g.detach())

    def _train_backward(e: dict) -> tuple[float, float]:
        """Backward L_H + λ mean_c L_G_c; stream components to limit peak VRAM."""
        sid = str(e["structure_id"])
        m.set_graph(Path(e["path"]).resolve())
        H = m.forward_H(pos=None)
        h_mid, dh_dr, mask_np = labels[sid]
        mask = torch.as_tensor(mask_np, device=H.device, dtype=torch.bool)
        t = torch.as_tensor(h_mid, device=H.device, dtype=H.dtype)
        loss_h = relfro(H[mask], t[mask])
        loss_h.backward()
        h_err = float(loss_h.detach())
        if lambda_d == 0.0 or dh_dr is None:
            return h_err, float("nan")
        atoms = [int(a) for a in e["probe_atoms"].tolist()]
        axes = [int(a) for a in manifest["axes"]]
        dh = torch.as_tensor(dh_dr, device=H.device, dtype=H.dtype)
        c_count = len(atoms) * len(axes)
        scale = float(lambda_d) / float(c_count)
        g_vals: list[float] = []
        k = 0
        for atom in atoms:
            for axis in axes:
                d_pred = model_component_fd(m, atom, int(axis), delta)
                loss_g_c = relfro(d_pred[mask], dh[k][mask])
                g_vals.append(float(loss_g_c.detach()))
                (scale * loss_g_c).backward()
                k += 1
        return h_err, float(sum(g_vals) / c_count)

    best_val = float("inf")
    history: list[dict] = []
    out_stem = _resolve(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        m.model.train()
        t0 = time.time()
        h_acc = 0.0
        d_acc = 0.0
        for e in split["train"]:
            optimizer.zero_grad()
            h_err, d_err = _train_backward(e)
            torch.nn.utils.clip_grad_norm_(m.model.parameters(), 1.0)
            optimizer.step()
            h_acc += h_err
            if not math.isnan(d_err):
                d_acc += d_err
        h_acc /= max(len(split["train"]), 1)
        d_acc = float("nan") if lambda_d == 0.0 else d_acc / max(len(split["train"]), 1)

        if epoch % args.eval_every == 0 or epoch == 1:
            m.model.eval()
            vh = [_structure_loss(e)[0] for e in split["val"]]
            val_h = float(np.mean(vh))
            if lambda_d == 0.0:
                val_d = float("nan")
                score = val_h
            else:
                vd = [_structure_loss(e)[1] for e in split["val"]]
                val_d = float(np.mean(vd))
                score = val_h + lambda_d * val_d
            meta_key = "g1_static" if lambda_d == 0.0 else "g1b"
            if score < best_val:
                best_val = score
                torch.save(
                    {
                        "state_dict": {k: v.detach().cpu() for k, v in m.model.state_dict().items()},
                        meta_key: {
                            "epoch": epoch, "val_h_rel_err": val_h, "val_d_rel_err": val_d,
                            "seed": args.seed, "lr": args.lr, "lambda_d": lambda_d,
                            "static_only": bool(lambda_d == 0.0),
                            "base_ckpt": str(paths.ckpt),
                        },
                    },
                    out_stem.with_name(out_stem.name + "_best.ckpt"),
                )
            if lambda_d == 0.0:
                print(
                    f"epoch {epoch:4d} | {time.time()-t0:5.1f}s | train H {h_acc:.4f} "
                    f"| val H {val_h:.4f} [static-only]",
                    flush=True,
                )
            else:
                print(
                    f"epoch {epoch:4d} | {time.time()-t0:5.1f}s | train H {h_acc:.4f} d {d_acc:.4f} "
                    f"| val H {val_h:.4f} d {val_d:.4f}",
                    flush=True,
                )
            history.append(
                {"epoch": epoch, "train_h": h_acc, "train_d": d_acc,
                 "val_h": val_h, "val_d": val_d}
            )
        else:
            if lambda_d == 0.0:
                print(
                    f"epoch {epoch:4d} | {time.time()-t0:5.1f}s | train H {h_acc:.4f} [static-only]",
                    flush=True,
                )
            else:
                print(
                    f"epoch {epoch:4d} | {time.time()-t0:5.1f}s | train H {h_acc:.4f} d {d_acc:.4f}",
                    flush=True,
                )
            history.append({"epoch": epoch, "train_h": h_acc, "train_d": d_acc})
        history_path = out_stem.with_name(out_stem.name + "_history.json")
        history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    # Final eval on the test fold set.
    m.model.eval()
    th = [_structure_loss(e)[0] for e in split["test"]]
    if lambda_d == 0.0:
        test_d = float("nan")
        print(
            f"\ntest (folds {sorted(folds_test)}): H rel-err {np.mean(th):.4f} [static-only]"
        )
    else:
        td = [_structure_loss(e)[1] for e in split["test"]]
        test_d = float(np.mean(td))
        print(
            f"\ntest (folds {sorted(folds_test)}): H rel-err {np.mean(th):.4f} "
            f"dH/dR rel-err {test_d:.4f}"
        )

    meta_key = "g1_static" if lambda_d == 0.0 else "g1b"
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in m.model.state_dict().items()},
            meta_key: {
                "epoch": epoch, "seed": args.seed, "lr": args.lr, "lambda_d": lambda_d,
                "static_only": bool(lambda_d == 0.0),
                "base_ckpt": str(paths.ckpt),
                "test_h_rel_err": float(np.mean(th)),
                "test_d_rel_err": test_d,
                "history": history,
            },
        },
        out_stem.with_name(out_stem.name + "_final.ckpt"),
    )
    print(f"wrote {out_stem}_final.ckpt (best_val {best_val:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
