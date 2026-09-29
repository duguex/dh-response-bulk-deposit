#!/usr/bin/env python3
"""Build a GaAs perfect-cell displacement cloud from Perf_chg0 for Track B bulk.

Writes:
  - NPZ graphs under ``--out-dir`` (torch_geometric Data; only ``pos`` changes)
  - ``config/dh_bulk_manifest.yaml`` with fixed probes and 5 geometry folds

Identity geometry is stem **Perf** pointing at the base ``Perf_chg0.npz``
(required by ``build_dh_label_cache._structure_rows``). Displaced stems are
``Bulk01``…``Bulk{N-1}``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DEFAULT_PROBES = [60, 28, 7, 32]
DEFAULT_CENTER = [10.682905197143555, 10.682905197143555, 10.682905197143555]


def _load_base_data(path: Path):
    blob = np.load(path, allow_pickle=True)
    graph_map = blob["graph"].item()
    if not isinstance(graph_map, dict) or 0 not in graph_map:
        raise ValueError(f"{path}: expected graph dict with key 0")
    return graph_map[0]


def _save_graph(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, graph=np.array({0: data}, dtype=object))


def _clone_with_pos(data, pos: np.ndarray):
    out = data.clone()
    out.pos = torch.as_tensor(np.asarray(pos, dtype=np.float64), dtype=data.pos.dtype)
    if hasattr(out, "total_energy"):
        out.total_energy = torch.as_tensor(0.0)
    return out


def _plan_displacements(
    n_atoms: int,
    n_geom: int,
    amplitude: float,
    seed: int,
    probes: list[int],
) -> list[dict]:
    if n_geom < 5:
        raise ValueError("n_geom must be >= 5 to fill five folds")
    rng = np.random.default_rng(seed)
    plans: list[dict] = [{"name": "Perf", "ops": [], "use_base_path": True}]
    pool = list(dict.fromkeys(list(probes) + list(range(n_atoms))))
    g = 1
    for atom in pool:
        if g >= n_geom:
            break
        for axis in range(3):
            if g >= n_geom:
                break
            sign = 1.0 if (g % 2 == 1) else -1.0
            amp = amplitude * float(rng.uniform(0.7, 1.0))
            plans.append(
                {
                    "name": f"Bulk{g:02d}",
                    "ops": [(int(atom), int(axis), float(sign * amp))],
                    "use_base_path": False,
                }
            )
            g += 1
    while g < n_geom:
        atoms = rng.choice(n_atoms, size=2, replace=False)
        ops = []
        for atom in atoms:
            axis = int(rng.integers(0, 3))
            sign = float(rng.choice([-1.0, 1.0]))
            amp = amplitude * float(rng.uniform(0.5, 1.0))
            ops.append((int(atom), axis, sign * amp))
        plans.append({"name": f"Bulk{g:02d}", "ops": ops, "use_base_path": False})
        g += 1
    return plans


def _apply_ops(pos0: np.ndarray, ops: list[tuple[int, int, float]]) -> np.ndarray:
    pos = np.array(pos0, dtype=np.float64, copy=True)
    for atom, axis, delta in ops:
        pos[int(atom), int(axis)] += float(delta)
    return pos


def _folds_for_stems(stems: list[str], n_folds: int = 5) -> dict[int, list[str]]:
    folds = {i: [] for i in range(n_folds)}
    for i, stem in enumerate(stems):
        folds[i % n_folds].append(stem)
    if any(len(v) == 0 for v in folds.values()):
        raise ValueError(f"need at least {n_folds} stems, got {len(stems)}")
    return folds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        default=_REPO / "data/npz/Perf_chg0.npz",
        help="base perfect-cell graph",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "data/bulk_perf",
        help="directory for displaced NPZ graphs",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_REPO / "config/dh_bulk_manifest.yaml",
    )
    parser.add_argument("--n-geom", type=int, default=20, help="including identity Perf")
    parser.add_argument("--amplitude", type=float, default=0.02, help="Å single-kick scale")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--probes", type=int, nargs=4, default=DEFAULT_PROBES)
    parser.add_argument("--center", type=float, nargs=3, default=DEFAULT_CENTER)
    args = parser.parse_args(argv)

    base_path = args.base.expanduser().resolve()
    if not base_path.is_file():
        print(f"ERROR: base graph missing: {base_path}", file=sys.stderr)
        return 2

    data0 = _load_base_data(base_path)
    pos0 = data0.pos.detach().cpu().numpy().astype(np.float64)
    plans = _plan_displacements(
        int(pos0.shape[0]),
        int(args.n_geom),
        float(args.amplitude),
        int(args.seed),
        list(args.probes),
    )

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = []
    stems: list[str] = []
    structures: dict[str, list[dict]] = {}

    for plan in plans:
        stem = str(plan["name"])
        stems.append(stem)
        if plan.get("use_base_path"):
            try:
                rel_path = base_path.relative_to(_REPO)
            except ValueError:
                rel_path = Path("data/npz/Perf_chg0.npz")
            abs_path = base_path
            max_disp = 0.0
        else:
            pos = _apply_ops(pos0, plan["ops"])
            data = _clone_with_pos(data0, pos)
            abs_path = out_dir / f"{stem}_chg0.npz"
            _save_graph(abs_path, data)
            try:
                rel_path = abs_path.relative_to(_REPO)
            except ValueError:
                rel_path = abs_path
            max_disp = float(np.max(np.linalg.norm(pos - pos0, axis=1)))

        structures[stem] = [
            {
                "charge": 0,
                "path": str(rel_path).replace("\\", "/"),
                "probe_atoms": [int(x) for x in args.probes],
                "defect_center": [float(x) for x in args.center],
                "defect_kind": "perfect",
            }
        ]
        meta_rows.append(
            {
                "stem": stem,
                "path": str(abs_path),
                "ops": [{"atom": a, "axis": ax, "delta_A": d} for a, ax, d in plan["ops"]],
                "max_atom_disp_A": max_disp,
            }
        )

    folds_map = _folds_for_stems(stems, n_folds=5)
    manifest = {
        "delta_angstrom": 0.001,
        "probe_atoms_per_structure": 4,
        "axes": [0, 1, 2],
        "seed": int(args.seed),
        "campaign": "track_b_bulk_perf_displacement_cloud_v1",
        "base_graph": str(base_path),
        "notes": (
            "Track B bulk (ADR-0006). Perfect-cell geometries only; charge=0. "
            "Stem Perf = identity Perf_chg0; Bulk** = displaced. "
            "Probes fixed to historical Perf_chg0 set."
        ),
        "folds": {str(k): {"stems": v} for k, v in sorted(folds_map.items())},
        "structures": structures,
    }

    man_path = args.manifest.expanduser().resolve()
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    meta_path = out_dir / "bulk_cloud_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "base": str(base_path),
                "n_geom": len(plans),
                "amplitude_A": float(args.amplitude),
                "seed": int(args.seed),
                "probes": [int(x) for x in args.probes],
                "center": [float(x) for x in args.center],
                "manifest": str(man_path),
                "geometries": meta_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    n_disp = sum(1 for r in meta_rows if r["stem"] != "Perf")
    print(f"wrote {n_disp} displaced graphs under {out_dir}")
    print(f"identity stem Perf -> {base_path}")
    print(f"wrote manifest {man_path}")
    print(f"wrote meta {meta_path}")
    print("folds:", {k: len(v) for k, v in folds_map.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
