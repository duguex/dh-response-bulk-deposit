#!/usr/bin/env python3
"""Track B downstream leg: Gamma band-window de/dR response, three arms.

For each test-fold bulk geometry and each (atom, axis) probe component:

  reference  generalized-eigenvalue FD of OpenMX (H+, S+) vs (H-, S-) extracted
             from the label-cache scfout pairs (read_openmx writes HS.json to
             its cwd; window defined per structure on the reference midpoint
             spectrum);
  model      generalized-eigenvalue FD of (H_gamma(pos±d), S_gamma(pos±d)) per
             arm checkpoint.

Observables per component: CBM gradient (singlet), VBM-triplet gradient sum,
each reported as model-vs-reference (MAE, median rel err, correlation, sign
agreement, RMS norm ratio). The VBM sum is gauge-free inside the degenerate
manifold; single triplet members are not.

Inference contract (validated 2026-09-06): packed forward_H <-> H_gamma dense
agree bit-for-bit through the packed relfro 0.00431; the same assembly on
reference blocks reproduces the G4 host gap 0.711 eV; S responds to pos.
Requires a read_openmx binary writing HS.json into its cwd (rebuild without
MPI: gcc read_openmx.c -lm).

Example:
  python scripts/diff_hamgnn/bulk_band_response_eval.py \
    --manifest config/dh_bulk_manifest_abs.yaml \
    --cache-root /mnt/shared/work/dh_response_bulk_labels_v1 \
    --ckpt-tag frozen=/mnt/shared/work/dh_response_g1a/g1a_embq_fix_r2.ckpt \
    --ckpt-tag h_only=/mnt/shared/work/dh_response_bulk_train/h_only_best.ckpt \
    --ckpt-tag joint=/mnt/shared/work/dh_response_bulk_train/joint_best.ckpt \
    --reader /tmp/read_openmx_standalone \
    --work-root /mnt/shared/work/dh_response_bulk_band_eval \
    --out docs/analysis/dh_response/bulk_band_response/eval.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402

_orig_torch_load = torch.load
torch.load = lambda *a, **k: _orig_torch_load(*a, **{**k, "weights_only": False})

BOHR_TO_ANG = 0.52917721092
NAO = 19
EV_A_PER_HA_BOHR = 51.422086  # 27.211386 eV/Ha / 0.529177 A/Bohr


def _extract_hs(reader: Path, scfout: Path, work_dir: Path) -> dict[str, np.ndarray]:
    """read_openmx once per scfout; cache the needed arrays as compact npz."""
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / "HS.npz"
    if out.exists():
        return dict(np.load(out))
    tmp = work_dir / ".run"
    tmp.mkdir(exist_ok=True)
    run = subprocess.run([str(reader), str(scfout)], cwd=tmp,
                         capture_output=True, text=True, timeout=600)
    raw = tmp / "HS.json"
    if run.returncode != 0 or not raw.exists():
        raise RuntimeError(f"read_openmx failed for {scfout}: {run.stderr[-300:]}")
    payload = json.loads(raw.read_text())
    keep: dict[str, np.ndarray] = {}
    for k in ("Hon", "Hoff", "Son", "Soff"):
        keep[k] = np.asarray(payload[k], dtype=np.float64)
    keep["edge_index"] = np.asarray(payload["edge_index"], dtype=np.int64)
    keep["cell_shift"] = np.asarray(payload["cell_shift"], dtype=np.int64)
    keep["pos"] = np.asarray(payload["pos"], dtype=np.float64)
    np.savez(out, **keep)
    return keep


def _dense_hs(hs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Dense Gamma H,S from reader blocks (rows = src = edge_index[0])."""
    from scripts.diff_hamgnn.openmx_hf_decomposition import _unwrap_blocks
    ei = hs["edge_index"]
    n_atoms = hs["Son"].shape[0]
    n_edges = ei.shape[1]
    Hon = _unwrap_blocks(hs["Hon"], n_atoms).reshape(n_atoms, NAO, NAO)
    Hoff = _unwrap_blocks(hs["Hoff"], n_edges).reshape(n_edges, NAO, NAO)
    Son = hs["Son"].reshape(n_atoms, NAO, NAO)
    Soff = hs["Soff"].reshape(n_edges, NAO, NAO)
    H = np.zeros((n_atoms * NAO, n_atoms * NAO))
    S = np.zeros_like(H)
    for a in range(n_atoms):
        s = slice(a * NAO, (a + 1) * NAO)
        H[s, s] += Hon[a]
        S[s, s] += Son[a]
    for e, (src, dst) in enumerate(ei.T):
        H[src * NAO:(src + 1) * NAO, dst * NAO:(dst + 1) * NAO] += Hoff[e]
        S[src * NAO:(src + 1) * NAO, dst * NAO:(dst + 1) * NAO] += Soff[e]
    return 0.5 * (H + H.T), 0.5 * (S + S.T)


def _window_from_reference(H_mid: np.ndarray, S_mid: np.ndarray, n_occ: int) -> list[int]:
    import scipy.linalg as sla
    ev = sla.eigh(H_mid, S_mid, eigvals_only=True)
    val = list(range(n_occ - 3, n_occ))
    assert ev[val].std() < 5e-4, f"VBM triplet not degenerate: {ev[val]}"
    assert ev[n_occ] > ev[n_occ - 1], "no gap at reference n_occ"
    return val + [n_occ]


def _arm_metrics(rows: list[dict]) -> dict:
    def stats(key_r: str, key_m: str) -> dict:
        r = np.array([x[key_r] for x in rows])
        q = np.array([x[key_m] for x in rows])
        rel = np.abs(q - r) / (np.abs(r) + 1e-12)
        return {
            "n": int(len(r)),
            "mae_eV_per_A": float(np.mean(np.abs(q - r))) * EV_A_PER_HA_BOHR,
            "median_rel": float(np.median(rel)),
            "corr": float(np.corrcoef(r, q)[0, 1]) if np.std(r) > 0 and np.std(q) > 0 else float("nan"),
            "sign_agree": float(np.mean(np.sign(r) == np.sign(q))),
            "norm_ratio_rms": float(np.sqrt(np.mean(q**2)) / np.sqrt(np.mean(r**2))),
        }
    return {"cbm": stats("ref_cbm", "mod_cbm"), "vsum": stats("ref_vsum", "mod_vsum"),
            "W_relerr_median": float(np.median([x["W_relerr"] for x in rows])),
            "W_relerr_p25": float(np.percentile([x["W_relerr"] for x in rows], 25)),
            "W_relerr_p75": float(np.percentile([x["W_relerr"] for x in rows], 75)),
            "noise_floor_median": float(np.median([x["noise_floor"] for x in rows]))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--ckpt-tag", action="append", required=True)
    ap.add_argument("--reader", type=Path, required=True)
    ap.add_argument("--work-root", type=Path, required=True)
    ap.add_argument("--hamgnn-root", type=Path,
                    default=Path("/home/duguex/defect_research/repos/HamGNN_v2.0"))
    ap.add_argument("--config", type=Path, default=_REPO / "config" / "config_zenodo_test.yaml")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--folds-test", nargs="+", type=int, default=[4])
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    import scipy.linalg as sla
    from scripts.diff_hamgnn import DifferentiableEdgeSProvider, DifferentiableHamGNN
    from scripts.diff_hamgnn.build_dh_label_cache import _load_manifest, _structure_rows
    from scripts.diff_hamgnn.openmx_hf_decomposition import decompose_dense_fd

    manifest, _ = _load_manifest(args.manifest.expanduser().resolve())
    delta = float(args.delta or manifest.get("delta_angstrom", 0.001))
    axes = [int(a) for a in manifest["axes"]]
    entries, fails = _structure_rows(manifest, args.manifest.expanduser().resolve(),
                                     load_supplied_structures=True)
    if fails:
        print("WARN structure failures:", fails, file=sys.stderr)
    test_entries = [e for e in entries if int(e["fold"]) in set(args.folds_test)]
    print(f"test structures: {[str(e['structure_id']) for e in test_entries]}", flush=True)
    cache_root = args.cache_root.expanduser().resolve()

    # ---- reference side: extract scfout pairs, dense H±/S± per component ----
    # Units: reader pos and model graph pos are both Bohr (validated: base
    # coordinates agree to 1e-5; lattice 21.3658 Bohr = 11.31 A = 2x a_GaAs).
    # The FD half-displacement is self-calibrated per component from the plus/
    # minus scfout positions (manifest delta_angstrom=0.001 A = 0.0018897 Bohr).
    ref: dict[str, dict] = {}
    for e in test_entries:
        sid = str(e["structure_id"])
        atoms = [int(a) for a in e["probe_atoms"].tolist()]
        pairs = [(a, ax) for a in atoms for ax in axes]
        comp: dict[tuple[int, int], dict] = {}
        pos_check = None
        for (atom, axis) in pairs:
            tag = f"{sid}_atom{atom:03d}_axis{axis}"
            hs_p = _extract_hs(args.reader, cache_root / "jobs" / tag / "plus" / f"{tag}_plus.scfout",
                               args.work_root / tag / "plus")
            hs_m = _extract_hs(args.reader, cache_root / "jobs" / tag / "minus" / f"{tag}_minus.scfout",
                               args.work_root / tag / "minus")
            Hp, Sp = _dense_hs(hs_p)
            Hm, Sm = _dense_hs(hs_m)
            delta_c = float(hs_p["pos"][atom, axis] - hs_m["pos"][atom, axis]) / 2.0
            if not (1e-5 < abs(delta_c) < 1e-1):
                raise RuntimeError(f"{tag}: implausible FD displacement {delta_c}")
            dec = decompose_dense_fd(Hp, Sp, Hm, Sm, delta_c,
                                     bands=None)  # full spectrum once; window applied later
            # window probe basis from the reference midpoint + exact-S dS term
            import scipy.linalg as sla
            H_mid = 0.5 * (Hp + Hm)
            S_mid = 0.5 * (Sp + Sm)
            _, C_all = sla.eigh(H_mid, S_mid)
            dS_dense = (Sp - Sm) / (2.0 * delta_c)
            comp[(atom, axis)] = {"Hp": Hp, "Sp": Sp, "Hm": Hm, "Sm": Sm,
                                  "delta_c": delta_c,
                                  "g_HF": dec["g_HF"], "g_eig_fd": dec["g_eig_fd"],
                                  "eps_mid": dec["eigenvalues_mid"],
                                  "C_all": C_all, "dS_dense": dS_dense}
            if pos_check is None:
                pos_check = {"hs_pos_plus": hs_p["pos"], "atom": atom, "axis": axis,
                             "delta_c": delta_c}
        ref[sid] = {"pairs": pairs, "comp": comp, "pos_check": pos_check}
        print(f"  extracted {sid}: {len(comp)} components", flush=True)

    # ---- model arms ----
    summary: dict = {"delta": delta, "test": [str(e["structure_id"]) for e in test_entries],
                     "geometry_check": {}, "arms": {}}
    for tag_expr in args.ckpt_tag:
        tag, ckpt = tag_expr.split("=", 1)
        # optional response normalization: name=path@scale multiplies the model's
        # FD dH by scale before projection. Incident 2026-09-13: joint-trained
        # arms carry a 1.889716x per-Angstrom-vs-per-Bohr response factor
        # (labels stored per-A, trainer FD per-Bohr); pass @0.52917721 to
        # normalize them. frozen was never derivative-trained -> keep @1.0.
        resp_scale = 1.0
        if "@" in ckpt:
            ckpt, sfx = ckpt.rsplit("@", 1)
            resp_scale = float(sfx)
        m = DifferentiableHamGNN.from_files(
            args.config, Path(ckpt), args.hamgnn_root, device=args.device,
            freeze_weights=True, emb_q_slots=24)
        arm_rows: list[dict] = []
        for e in test_entries:
            sid = str(e["structure_id"])
            m.set_graph(Path(e["path"]).resolve())
            m.set_s_provider(DifferentiableEdgeSProvider(m.data, nao=m.nao))
            n_occ = m.n_valence_electrons() // 2
            comp = ref[sid]["comp"]
            pc = ref[sid]["pos_check"]
            if sid not in summary["geometry_check"]:
                hs_ang = pc["hs_pos_plus"] * BOHR_TO_ANG
                scale_ref = np.linalg.norm(m.data.pos.cpu().numpy()) / np.linalg.norm(pc["hs_pos_plus"])
                if abs(scale_ref - 1.0) > 0.01:
                    raise RuntimeError(f"unit mismatch model pos vs reader pos: scale={scale_ref}")
                expected = m.data.pos.cpu().numpy().astype(np.float64).copy()
                expected[pc["atom"], pc["axis"]] += pc["delta_c"]
                summary["geometry_check"][sid] = {
                    "max_abs_diff_bohr": float(np.abs(hs_ang / BOHR_TO_ANG - expected).max()),
                    "delta_c_bohr": float(pc["delta_c"]),
                }
            first = next(iter(comp.values()))
            window = _window_from_reference(
                0.5 * (first["Hp"] + first["Hm"]), 0.5 * (first["Sp"] + first["Sm"]), n_occ)
            val_idx, cond_idx = window[:3], window[3]
            from scripts.diff_hamgnn.evaluate_dh_response import model_component_fd
            from scripts.diff_hamgnn.assemble import assemble_gamma, split_hon_hoff
            with torch.no_grad():
                for (atom, axis), mats in comp.items():
                    C = mats["C_all"][:, window]                      # (N, 4) probe basis
                    eps_w = mats["eps_mid"][window]
                    dS_w = np.einsum("in,ij,jn->n", C, mats["dS_dense"], C).real
                    dH_m = model_component_fd(m, atom, axis, mats["delta_c"])
                    if resp_scale != 1.0:
                        dH_m = dH_m * resp_scale
                    Hon, Hoff = split_hon_hoff(dH_m.cpu(), m.n_atoms)
                    dH_dense = assemble_gamma(Hon, Hoff, m.edge_index.cpu(), m.nao).numpy().astype(np.float64)
                    gH = np.einsum("in,ij,jn->n", C, dH_dense, C).real
                    g_mod = gH - eps_w * dS_w
                    d_c = mats["delta_c"]
                    mod_cbm = float(g_mod[3])
                    mod_vsum = float(g_mod[:3].sum())
                    r_cbm = float(mats["g_HF"][cond_idx])
                    r_vsum = float(mats["g_HF"][val_idx].sum())
                    # splitting-pattern check: 3x3 triplet response matrix W = C^T dH C
                    C3 = mats["C_all"][:, val_idx]
                    dH_ref = (mats["Hp"] - mats["Hm"]) / (2 * d_c)
                    W_ref = C3.T @ dH_ref @ C3
                    W_mod = C3.T @ dH_dense @ C3
                    w_rel = float(np.linalg.norm(W_mod - W_ref) / np.linalg.norm(W_ref))
                    # float32 noise floor: same component at 2x displacement
                    dH_m2 = model_component_fd(m, atom, axis, 2 * d_c)
                    if resp_scale != 1.0:
                        dH_m2 = dH_m2 * resp_scale
                    Hon2, Hoff2 = split_hon_hoff(dH_m2.cpu(), m.n_atoms)
                    dH_dense2 = assemble_gamma(Hon2, Hoff2, m.edge_index.cpu(), m.nao).numpy().astype(np.float64)
                    W_mod2 = C3.T @ dH_dense2 @ C3
                    noise = float(np.linalg.norm(W_mod2 - W_mod) / np.linalg.norm(W_ref))
                    arm_rows.append({"structure_id": sid, "atom": atom, "axis": axis,
                                     "ref_cbm": r_cbm, "mod_cbm": mod_cbm,
                                     "ref_vsum": r_vsum, "mod_vsum": mod_vsum,
                                     "W_relerr": w_rel, "noise_floor": noise,
                                     "g_ref_win": [float(x) for x in mats["g_HF"][window]],
                                     "g_mod_win": [float(x) for x in g_mod]})
        summary["arms"][tag] = {"metrics": _arm_metrics(arm_rows), "rows": arm_rows,
                                "ckpt": ckpt, "n_occ": n_occ, "response_scale": resp_scale}
        print(f"[{tag}] " + json.dumps(summary["arms"][tag]["metrics"]), flush=True)
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
