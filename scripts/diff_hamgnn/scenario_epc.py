#!/usr/bin/env python3
# Archive note (frozen 2026-09-25; as-run copy of
#   /mnt/shared/work/dh_response_bulk_train/phase3/scenario_epc.py).
# Backs Table 3 (band-edge deformation coupling) of the Track B manuscript:
#   W = C^T dH C in eV/A for the valence triplet plus the CBM singlet, reported
#   as triplet trace/3 (hydrostatic-like), shear-like off-diagonal Frobenius
#   norm, and the CBM diagonal, each times u0 (zone-centre LO zero-point
#   amplitude, w_LO = 8.75 THz).
# Derivative-trained arms carry the response normalization RESP_SCALE =
#   0.52917721 (ADR-0012 unit incident); the frozen arm uses 1.0.
# REPO/WORK/V1/CKPTS below are the as-run absolute paths; the caches they read
#   (dh_response_bulk_band_eval2, dh_response_bulk_labels_v1) are deposited with
#   the data record.
"""Consuming-scenario table: Gamma band-edge deformation coupling of GaAs.

For each test-fold component (structure, probe atom, axis), the DFT
reference and each model arm give the band-edge response in the
reference gauge. We report the physical coupling directly, in eV/A and
as a level shift at the zone-centre LO zero-point amplitude
u0 = sqrt(hbar/(2 M w_LO)), w_LO(GaAs) = 8.75 THz:
  - VBM triplet: hydrostatic-like trace/3 and shear-like off-diagonal
    Frobenius norm of the 3x3 response (gauge-free);
  - CBM singlet diagonal.
This is the electronic factor of the electron-phonon coupling: the
object consumed by EPC and carrier-scattering algebra.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/duguex/defect_research/repos/slurm-4090-guide/.worktrees/dh-response-paper")
sys.path.insert(0, str(REPO))
import torch  # noqa: E402
_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})

import scipy.linalg as sla  # noqa: E402
from scripts.diff_hamgnn import DifferentiableHamGNN  # noqa: E402
from scripts.diff_hamgnn.bulk_band_response_eval import _extract_hs, _dense_hs  # noqa: E402
from scripts.diff_hamgnn.assemble import assemble_gamma, split_hon_hoff  # noqa: E402
from scripts.diff_hamgnn.evaluate_dh_response import model_component_fd  # noqa: E402

NAO = 19
EV_A_PER_HA_BOHR = 51.422086                 # eV/A per Ha/Bohr
HBAR_J = 1.054571817e-34                     # hbar in J*s
M_U = 1.66053906660e-27
MASS_U = {"Ga": 69.723, "As": 74.922}
W_LO_THZ = 8.75
RESP_SCALE = {"frozen": 1.0, "joint": 0.52917721, "calib": 0.52917721}

WORK = Path("/mnt/shared/work/dh_response_bulk_band_eval2")
V1 = Path("/mnt/shared/work/dh_response_bulk_labels_v1/jobs")
CKPTS = {
    "frozen": "/mnt/shared/work/dh_response_g1a/g1a_embq_fix_r2.ckpt",
    "joint": "/mnt/shared/work/dh_response_bulk_train/v2c/joint_best.ckpt",
    "calib": "/mnt/shared/work/dh_response_bulk_train/phase3/lam10_cont80/lam10d_final.ckpt",
}
READER = Path("/tmp/read_openmx_standalone")
HAMGNN = "/home/duguex/defect_research/repos/HamGNN_v2.0"
CONFIG = REPO / "config" / "config_zenodo_test.yaml"

COMPS = [  # component, atom (0-based), axis
    ("Bulk04_chg0_atom060_axis1", 60, 1),
    ("Bulk19_chg0_atom032_axis1", 32, 1),
    ("Bulk09_chg0_atom028_axis2", 28, 2),
    ("Bulk14_chg0_atom007_axis0", 7, 0),
]


def u0(species: str) -> float:
    w = 2.0 * np.pi * W_LO_THZ * 1e12
    return np.sqrt(HBAR_J / (2.0 * MASS_U[species] * M_U * w)) * 1e10  # Angstrom


def atom_species(npz_path: Path, atom: int) -> str:
    z = np.load(npz_path, allow_pickle=True)
    for key in ("atomic_numbers", "z_table", "numbers"):
        if key in z:
            n = int(np.asarray(z[key]).flatten()[atom])
            return "Ga" if n == 31 else ("As" if n == 33 else f"Z{n}")
    raise SystemExit(f"no atomic numbers in {npz_path}: {list(z.keys())}")


def load_ref(comp: str, atom: int, axis: int) -> tuple[np.ndarray, np.ndarray, float]:
    hp = _extract_hs(READER, V1 / comp / "plus" / f"{comp}_plus.scfout", WORK / comp / "plus")
    hm = _extract_hs(READER, V1 / comp / "minus" / f"{comp}_minus.scfout", WORK / comp / "minus")
    Hp, Sp = _dense_hs(hp)
    Hm, Sm = _dense_hs(hm)
    dc = float(hp["pos"][atom, axis] - hm["pos"][atom, axis]) / 2.0
    dH = (Hp - Hm) / (2.0 * dc)
    Hmid, Smid = 0.5 * (Hp + Hm), 0.5 * (Sp + Sm)
    ev, C = sla.eigh(Hmid, Smid)
    n_occ = int(np.where(ev > -0.5)[0][0])          # crude; replaced below if needed
    return dH, C, dc


def model_dH(m, struct_npz: Path, atom: int, axis: int, dc: float) -> np.ndarray:
    m.set_graph(struct_npz)
    with torch.no_grad():
        d = model_component_fd(m, atom, axis, dc).cpu().numpy()
    d = assemble_gamma(*split_hon_hoff(torch.from_numpy(d), m.n_atoms),
                       m.edge_index.cpu(), NAO).numpy().astype(np.float64)
    return d * EV_A_PER_HA_BOHR   # Ha/Bohr -> eV/A


def main() -> None:
    # one model carries the electron count (consistent across ckpts)
    m0 = DifferentiableHamGNN.from_files(CONFIG, Path(CKPTS["frozen"]), HAMGNN,
                                         device="cuda:0", freeze_weights=True, emb_q_slots=24)
    m0.set_graph(REPO / "data" / "npz" / "Bulk04_chg0.npz")
    n_occ = m0.n_valence_electrons() // 2
    z_nums = np.asarray(m0.data.z.tolist())
    print(f"n_occ = {n_occ}")
    for comp, atom, axis in COMPS:
        stru = REPO / "data" / "npz" / f"{comp[:comp.index('_chg0')]}_chg0.npz"
        n = int(z_nums[atom])
        species = "Ga" if n == 31 else ("As" if n == 33 else f"Z{n}")
        u = u0(species)
        dH, C, dc = load_ref(comp, atom, axis)
        assert np.isfinite(dH).all()
        window = list(range(n_occ - 3, n_occ)) + [n_occ]
        Cw = C[:, window]
        Wr = (Cw.T @ dH @ Cw) * EV_A_PER_HA_BOHR        # eV/A
        W3r = Wr[:3, :3]
        tr_r = W3r.trace() / 3.0
        sh_r = np.linalg.norm(W3r - np.diag(np.diag(W3r)), ord="fro")
        cb_r = Wr[3, 3]
        print(f"\n[{comp}]  {species}  u0={u:.3f} A   (DFT: trace/3={tr_r:8.3f} eV/A, "
              f"shear={sh_r:8.3f} eV/A, CBM={cb_r:8.3f} eV/A)")
        for name, ck in CKPTS.items():
            m = DifferentiableHamGNN.from_files(CONFIG, Path(ck), HAMGNN, device="cuda:0",
                                                freeze_weights=True, emb_q_slots=24)
            dHm = model_dH(m, stru, atom, axis, dc) * RESP_SCALE[name]  # response normalization (ADR-0012)
            Wm = Cw.T @ dHm @ Cw
            W3 = Wm[:3, :3]
            tr, sh, cb = W3.trace() / 3.0, np.linalg.norm(W3 - np.diag(np.diag(W3)), ord="fro"), Wm[3, 3]
            # level shifts at LO zero-point amplitude, in meV
            dt, ds, dc_ = (tr - tr_r) * u * 1e3, (sh - sh_r) * u * 1e3, (cb - cb_r) * u * 1e3
            print(f"  {name:7s}: trace {tr:8.3f} eV/A (ref {tr_r:8.3f}, +{dt:7.3f} meV) | "
                  f"shear {sh:8.3f} (ref {sh_r:8.3f}, +{ds:7.3f} meV) | "
                  f"CBM {cb:8.3f} (ref {cb_r:8.3f}, +{dc_:7.3f} meV)")


if __name__ == "__main__":
    main()
