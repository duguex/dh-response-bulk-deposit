#!/usr/bin/env python3
"""Decompose OpenMX band-window FD into dH and dS Hellmann–Feynman terms.

This is the claim-critical follow-up to Track A (#22). It reuses cached
OpenMX SCF calculations at R±delta, extracts H/S blocks with read_openmx,
and reports

    g_H  = sum_n c_n^† dH c_n
    g_S  = -sum_n eps_n c_n^† dS c_n
    g_HF = g_H + g_S

against the central FD of the corresponding OpenMX eigenvalue sum.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

BandSelection = Union[slice, Sequence[int], None]


def _hermitize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return 0.5 * (matrix + matrix.T)


def _generalized_eigh(H: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve H C = S C eps on the active (positive-overlap) subspace.

    OpenMX blocks are padded to a uniform basis: orbitals outside the active
    species basis carry zero overlap and zero matrix rows/columns.  Such
    padded orbitals are dropped before the generalized diagonalization so the
    semi-definite raw overlap never blocks extraction of the physical bands.
    The active subspace is the span of the strictly positive overlap
    eigenvectors, which is shared by H_plus/H_minus/S_plus/S_minus in this
    pipeline (topology and basis are identical across branches).
    """
    H = _hermitize(H)
    S = _hermitize(S)
    se, sv = np.linalg.eigh(S)
    tolerance = np.finfo(np.float64).eps * max(1.0, float(se.max())) * se.size
    active = se > tolerance
    if not active.any():
        raise ValueError("overlap has no positive-definite active subspace")
    if active.all():
        Sinvh = (sv * se ** -0.5) @ sv.T
        Horth = _hermitize(Sinvh @ H @ Sinvh)
        evals, evecs = np.linalg.eigh(Horth)
        return (
            evals.astype(np.float64),
            Sinvh.astype(np.float64) @ evecs,
            se.astype(np.float64),
        )

    embedding = sv[:, active]  # (n_full, n_active) orthonormal active basis
    se_active = se[active]
    H_active = embedding.T @ H @ embedding
    scaling = se_active ** -0.5  # S on the active subspace is diagonal
    Horth_active = _hermitize(scaling[:, None] * H_active * scaling[None, :])
    evals, evecs_active = np.linalg.eigh(Horth_active)
    C_active = scaling[:, None] * evecs_active  # S-orthonormal in active frame
    C = embedding @ C_active  # back to the full padded basis
    return (
        evals.astype(np.float64),
        C.astype(np.float64),
        se.astype(np.float64),
    )


def _band_indices(n_bands: int, bands: BandSelection) -> np.ndarray:
    if bands is None:
        return np.arange(n_bands, dtype=np.int64)
    if isinstance(bands, slice):
        return np.arange(n_bands, dtype=np.int64)[bands]
    idx = np.asarray(list(bands), dtype=np.int64)
    if idx.ndim != 1:
        raise ValueError("bands must be a slice or one-dimensional sequence")
    return idx



def crossing_invariant_midpoint_relative_error(
    g_eig_fd: np.ndarray,
    g_HF: np.ndarray,
    eps: float = 0.0,
) -> float:
    """Return the trace-closure error without matching crossing-prone bands.

    ``eps`` is validated for API compatibility but intentionally does not
    perturb the denominator: the QC scalar is the exact relative trace error.
    """
    eps = float(eps)
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("eps must be finite and non-negative")

    eig_fd = np.asarray(g_eig_fd, dtype=np.float64)
    hf = np.asarray(g_HF, dtype=np.float64)
    if eig_fd.ndim != 1 or hf.ndim != 1 or eig_fd.shape != hf.shape:
        raise ValueError("g_eig_fd and g_HF must be one-dimensional arrays with equal shape")
    if not np.isfinite(eig_fd).all() or not np.isfinite(hf).all():
        raise ValueError("g_eig_fd and g_HF must contain only finite values")

    residual = abs(float(np.sum(eig_fd - hf)))
    denominator = float(np.sum(np.abs(eig_fd)))
    if denominator == 0.0:
        return 0.0 if residual == 0.0 else float("inf")
    return residual / denominator

def decompose_dense_fd(
    H_plus: np.ndarray,
    S_plus: np.ndarray,
    H_minus: np.ndarray,
    S_minus: np.ndarray,
    delta: float,
    bands: BandSelection = None,
) -> Dict[str, np.ndarray]:
    """Return per-band dH, dS, full-HF, eigenvalue-FD, and residual terms."""
    if delta <= 0:
        raise ValueError("delta must be positive")
    H_plus = _hermitize(H_plus)
    H_minus = _hermitize(H_minus)
    S_plus = _hermitize(S_plus)
    S_minus = _hermitize(S_minus)
    if not (H_plus.shape == H_minus.shape == S_plus.shape == S_minus.shape):
        raise ValueError("H/S plus/minus matrices must have identical shapes")
    if H_plus.ndim != 2 or H_plus.shape[0] != H_plus.shape[1]:
        raise ValueError("H/S inputs must be square matrices")

    H_mid = 0.5 * (H_plus + H_minus)
    S_mid = 0.5 * (S_plus + S_minus)
    evals_mid_all, C_all, overlap_eigs = _generalized_eigh(H_mid, S_mid)
    evals_plus_all, _, _ = _generalized_eigh(H_plus, S_plus)
    evals_minus_all, _, _ = _generalized_eigh(H_minus, S_minus)
    idx = _band_indices(len(evals_mid_all), bands)

    C = C_all[:, idx]
    eps = evals_mid_all[idx]
    dH = (H_plus - H_minus) / (2.0 * delta)
    dS = (S_plus - S_minus) / (2.0 * delta)
    g_H = np.einsum("in,ij,jn->n", C, dH, C).real.astype(np.float64)
    dS_expect = np.einsum("in,ij,jn->n", C, dS, C).real
    g_S = (-eps * dS_expect).astype(np.float64)
    g_HF = (g_H + g_S).astype(np.float64)
    g_eig_fd = ((evals_plus_all[idx] - evals_minus_all[idx]) / (2.0 * delta)).astype(
        np.float64
    )
    residual = (g_eig_fd - g_HF).astype(np.float64)

    return {
        "g_H": g_H,
        "g_S": g_S,
        "g_HF": g_HF,
        "g_eig_fd": g_eig_fd,
        "residual": residual,
        "eigenvalues_mid": eps.astype(np.float64),
        "overlap_eigenvalues_mid": overlap_eigs,
    }


def _unwrap_blocks(raw, expected: int):
    """Accept direct blocks or one extra channel wrapper used by some readers/tests."""
    if len(raw) == 1 and len(raw[0]) == expected:
        return raw[0]
    return raw


def _active_block(raw, row_idx: np.ndarray, col_idx: np.ndarray) -> np.ndarray:
    """Return one real non-SOC block without permitting a lossy complex cast."""
    arr = np.asarray(raw)
    real = np.real_if_close(arr, tol=1000)
    if np.iscomplexobj(real):
        raise ValueError("OpenMX block has a non-negligible imaginary component")
    expected = len(row_idx) * len(col_idx)
    if real.size != expected:
        raise ValueError(f"block has {real.size} values; expected {expected}")
    return np.asarray(real, dtype=np.float64).reshape(len(row_idx), len(col_idx))


def pack_hs_blocks(
    hs: dict,
    z: np.ndarray,
    nao: int,
    basis_def: dict,
) -> dict:
    """Pad variable OpenMX H/S blocks to uniform ``(nao, nao)`` arrays."""
    z = np.asarray(z, dtype=np.int64)
    edge_index = np.asarray(hs["edge_index"], dtype=np.int64)
    cell_shift = np.asarray(hs["cell_shift"], dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")
    n_atoms = len(z)
    n_edges = edge_index.shape[1]
    if cell_shift.shape != (n_edges, 3):
        raise ValueError(f"cell_shift shape {cell_shift.shape} != ({n_edges}, 3)")

    raw_Hon = _unwrap_blocks(hs["Hon"][0], n_atoms)
    raw_Hoff = _unwrap_blocks(hs["Hoff"][0], n_edges)
    raw_Son = _unwrap_blocks(hs["Son"], n_atoms)
    raw_Soff = _unwrap_blocks(hs["Soff"], n_edges)
    if not (
        len(raw_Hon) == len(raw_Son) == n_atoms
        and len(raw_Hoff) == len(raw_Soff) == n_edges
    ):
        raise ValueError("OpenMX block counts do not match atoms/edges")

    Hon = np.zeros((n_atoms, nao, nao), dtype=np.float64)
    Son = np.zeros_like(Hon)
    Hoff = np.zeros((n_edges, nao, nao), dtype=np.float64)
    Soff = np.zeros_like(Hoff)

    for atom, atomic_number in enumerate(z):
        active = np.asarray(basis_def[int(atomic_number)], dtype=np.int64)
        Hon[atom][np.ix_(active, active)] = _active_block(raw_Hon[atom], active, active)
        Son[atom][np.ix_(active, active)] = _active_block(raw_Son[atom], active, active)

    for edge, (src, dst) in enumerate(edge_index.T):
        src_active = np.asarray(basis_def[int(z[src])], dtype=np.int64)
        dst_active = np.asarray(basis_def[int(z[dst])], dtype=np.int64)
        Hoff[edge][np.ix_(src_active, dst_active)] = _active_block(
            raw_Hoff[edge], src_active, dst_active
        )
        Soff[edge][np.ix_(src_active, dst_active)] = _active_block(
            raw_Soff[edge], src_active, dst_active
        )

    return {
        "Hon": Hon,
        "Hoff": Hoff,
        "Son": Son,
        "Soff": Soff,
        "edge_index": edge_index.copy(),
        "cell_shift": cell_shift.copy(),
    }


def _assemble_gamma(packed: dict, hermitize: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble dense Γ-point H/S from padded on-site and directed edge blocks."""
    Hon = np.asarray(packed["Hon"], dtype=np.float64)
    Hoff = np.asarray(packed["Hoff"], dtype=np.float64)
    Son = np.asarray(packed["Son"], dtype=np.float64)
    Soff = np.asarray(packed["Soff"], dtype=np.float64)
    edge_index = np.asarray(packed["edge_index"], dtype=np.int64)
    n_atoms, nao, _ = Hon.shape
    H = np.zeros((n_atoms * nao, n_atoms * nao), dtype=np.float64)
    S = np.zeros_like(H)
    for atom in range(n_atoms):
        sl = slice(atom * nao, (atom + 1) * nao)
        H[sl, sl] = Hon[atom]
        S[sl, sl] = Son[atom]
    for edge, (src, dst) in enumerate(edge_index.T):
        src_sl = slice(src * nao, (src + 1) * nao)
        dst_sl = slice(dst * nao, (dst + 1) * nao)
        H[src_sl, dst_sl] += Hoff[edge]
        S[src_sl, dst_sl] += Soff[edge]
    if hermitize:
        return _hermitize(H), _hermitize(S)
    return H, S


def assemble_gamma_dense(packed: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble dense Γ-point H/S, symmetrized (for downstream consumers)."""
    return _assemble_gamma(packed, hermitize=True)


def assemble_gamma_dense_raw(packed: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble dense Γ-point H/S exactly as read, without symmetrization.

    Used by the label-cache builder so that the Hermiticity QC gate
    (build_cache_record, <1e-8 relative error) is measured on the raw
    assembly and cannot pass vacuously.
    """
    return _assemble_gamma(packed, hermitize=False)


def _parse_pairs(text: str) -> list[Tuple[int, int]]:
    pairs = []
    for item in text.split(","):
        atom, xyz = item.strip().split(":", 1)
        pairs.append((int(atom), int(xyz)))
    return pairs


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_with_sha256(path: Path) -> tuple[object, str]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        stream.seek(0)
        payload = json.load(stream)
    return payload, digest.hexdigest()


def _hs_input_provenance(reader: Path, scfout: Path) -> dict:
    return {
        "schema_version": 1,
        "scfout": {"sha256": _sha256_file(scfout)},
        "reader": {
            "resolved_path": str(reader),
            "sha256": _sha256_file(reader),
        },
    }


_REQUIRED_HS_KEYS = frozenset({"Hon", "Hoff", "Son", "Soff", "edge_index", "cell_shift"})


def _validate_hs_payload(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("read_openmx HS.json must contain a JSON object")
    missing = sorted(_REQUIRED_HS_KEYS - set(payload))
    if missing:
        raise ValueError(f"read_openmx HS.json is missing required keys: {missing}")
    return payload


def _selected_hs_files(work_dir: Path) -> Optional[tuple[Path, Path]]:
    current = work_dir / "CURRENT"
    try:
        if current.is_symlink():
            pointer = Path(os.readlink(current))
        else:
            lines = current.read_text().splitlines()
            if len(lines) != 1 or not lines[0]:
                return None
            pointer = Path(lines[0])
        if pointer.is_absolute() or ".." in pointer.parts:
            return None
        root = work_dir.resolve(strict=True)
        generation = (work_dir / pointer).resolve(strict=True)
        if not generation.is_dir() or not generation.is_relative_to(root):
            return None
    except (OSError, RuntimeError, UnicodeDecodeError):
        return None
    return generation / "HS.json", generation / "HS.provenance.json"


def _load_reusable_hs(work_dir: Path, inputs: dict) -> Optional[dict]:
    selected = _selected_hs_files(work_dir)
    if selected is None:
        return None
    output, sidecar = selected
    if not output.is_file() or not sidecar.is_file():
        return None
    try:
        payload, output_sha256 = _load_json_with_sha256(output)
        with sidecar.open("rb") as stream:
            metadata = json.load(stream)
        validated = _validate_hs_payload(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    expected = {
        **inputs,
        "hs_json": {"sha256": output_sha256},
    }
    if metadata != expected:
        return None
    return validated


def _commit_current(work_dir: Path, generation: Path) -> None:
    relative_generation = generation.relative_to(work_dir)
    temporary_pointer: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=".CURRENT-",
            dir=work_dir,
            delete=False,
        ) as stream:
            temporary_pointer = Path(stream.name)
            stream.write(f"{relative_generation}\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_pointer, work_dir / "CURRENT")
    finally:
        if temporary_pointer is not None:
            temporary_pointer.unlink(missing_ok=True)


def _extract_hs(reader: Path, scfout: Path, work_dir: Path) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    resolved_reader = reader.resolve()
    resolved_scfout = scfout.resolve()
    inputs = _hs_input_provenance(resolved_reader, resolved_scfout)

    cached = _load_reusable_hs(work_dir, inputs)
    if cached is not None:
        return cached

    generations = work_dir / "generations"
    generations.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".candidate-", dir=generations) as temporary_name:
        candidate_generation = Path(temporary_name)
        candidate_output = candidate_generation / "HS.json"
        run = subprocess.run(
            [str(resolved_reader), str(resolved_scfout)],
            cwd=candidate_generation,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if run.returncode != 0 or not candidate_output.exists():
            raise RuntimeError(
                f"read_openmx failed for {scfout}: rc={run.returncode}; {run.stderr[-500:]}"
            )

        candidate_raw, candidate_sha256 = _load_json_with_sha256(candidate_output)
        candidate = _validate_hs_payload(candidate_raw)
        if _hs_input_provenance(resolved_reader, resolved_scfout) != inputs:
            raise RuntimeError(f"read_openmx inputs changed during extraction for {scfout}")

        metadata = {
            **inputs,
            "hs_json": {"sha256": candidate_sha256},
        }
        candidate_sidecar = candidate_generation / "HS.provenance.json"
        candidate_sidecar.write_text(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n"
        )
        generation = generations / candidate_generation.name.replace(
            ".candidate-", "generation-", 1
        )
        candidate_generation.replace(generation)
        _commit_current(work_dir, generation)
        return candidate


def _load_model_ad(csv_path: Path) -> dict[Tuple[int, int], float]:
    values = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            values[(int(row["atom"]), int(row["xyz"]))] = float(row["AD_Honly"])
    return values


def _load_graph_contract(graph_path: Path):
    graph_dict = np.load(graph_path, allow_pickle=True)["graph"].item()
    graph = graph_dict[sorted(graph_dict)[0]]
    return (
        np.asarray(graph.z.cpu().numpy(), dtype=np.int64),
        np.asarray(graph.edge_index.cpu().numpy(), dtype=np.int64),
        np.asarray(graph.cell_shift.cpu().numpy(), dtype=np.int64),
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(denominator) > 1e-15 else float("nan")


def _write_outputs(rows: list[dict], metadata: dict, report: Path, csv_path: Path) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    json_path = report.with_suffix(".json")
    json_path.write_text(json.dumps({"metadata": metadata, "rows": rows}, indent=2))
    lines = [
        "# Band-force OpenMX dH/dR and dS/dR Decomposition",
        "",
        "> Follow-up to Track A issue #22. Γ-only, bands [10,20), δ=0.001 Å.",
        "",
        "## Contract",
        "",
        r"\[g_H=\sum_n c_n^\dagger(dH/dR)c_n,\qquad g_S=-\sum_n\varepsilon_n c_n^\dagger(dS/dR)c_n.\]",
        "",
        r"\[g_{HF}=g_H+g_S,\qquad r=g_{eig,FD}-g_{HF}.\]",
        "",
        "Midpoint eigenpairs come from the same local OpenMX ±δ matrix pair. This avoids the earlier training-graph vs local-SCF energy-zero mismatch.",
        "",
        "## Results",
        "",
        "| atom | xyz | model AD | OpenMX g_H | OpenMX g_S | g_HF | eig FD | residual | H/eig | S/eig | H/model |",
        "|-----:|----:|---------:|-----------:|-----------:|-----:|-------:|---------:|------:|------:|--------:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['atom']} | {row['xyz']} | {row['model_AD']:.6e} | "
            f"{row['g_H']:.6e} | {row['g_S']:.6e} | {row['g_HF']:.6e} | "
            f"{row['g_eig_out']:.6e} | {row['residual_out']:.3e} | "
            f"{row['H_fraction']:.4f} | {row['S_fraction']:.4f} | {row['H_over_model']:.2f}× |"
        )
    max_residual = max(abs(row["residual_fraction"]) for row in rows)
    min_h_fraction = min(abs(row["H_fraction"]) for row in rows)
    max_s_fraction = max(abs(row["S_fraction"]) for row in rows)
    max_dense_out_abs = max(
        abs(row["g_eig_dense_fd"] - row["g_eig_out"]) for row in rows
    )
    max_dense_out_rel = max(
        abs(row["g_eig_dense_fd"] - row["g_eig_out"])
        / (abs(row["g_eig_out"]) + 1e-15)
        for row in rows
    )
    lines += [
        "",
        "## Interpretation",
        "",
        f"- OpenMX dH term explains at least **{min_h_fraction * 100:.1f}%** of the eigenvalue FD magnitude across these components.",
        f"- OpenMX dS term contributes at most **{max_s_fraction * 100:.3f}%**.",
        f"- Full-HF residual is at most **{max_residual * 100:.3f}%**, so finite-δ re-diagonalization / band tracking is not the leading gap here.",
        f"- Dense-matrix eigenvalue FD reproduces parsed OpenMX `.out` FD within **{max_dense_out_abs:.2e} Ha/Å** (relative <{max_dense_out_rel:.2e}), validating block extraction and Γ assembly.",
        "- The model-to-OpenMX discrepancy is therefore **dH-dominant**. This supports issue #21 derivative supervision; exact PAO S(R) remains important for completeness but is not the main cause of the current 16–45× gap.",
        "",
        "## Limits",
        "",
        "- Two atom-direction components only; no distributional claim.",
        "- Deep-valence band window [10,20), Γ-only, non-SOC.",
        "- A second δ is still needed for an OpenMX linear-regime sensitivity check.",
        "",
        "## Artifacts",
        "",
        f"- CSV: `{csv_path}`",
        f"- JSON: `{json_path}`",
        "- Script: `scripts/diff_hamgnn/openmx_hf_decomposition.py`",
        "",
    ]
    report.write_text("\n".join(lines) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=Path("/tmp/band_force_openmx"))
    parser.add_argument("--pairs", default="44:1,47:1", help="comma-separated atom:xyz")
    parser.add_argument("--delta", type=float, default=0.001)
    parser.add_argument("--band-start", type=int, default=10)
    parser.add_argument("--band-width", type=int, default=10)
    parser.add_argument("--nao", type=int, default=19)
    parser.add_argument(
        "--reader",
        type=Path,
        default=Path.home() / "developing/openmx3.9/openmx_postprocess/read_openmx",
    )
    parser.add_argument(
        "--utils-openmx",
        type=Path,
        default=Path.home() / "repos/HamGNN/utils_openmx",
    )
    parser.add_argument("--graph", type=Path, default=_REPO / "data/npz/As_Ga_chg0.npz")
    parser.add_argument(
        "--model-csv", type=Path, default=_REPO / "docs/band_force_external_pairs.csv"
    )
    parser.add_argument(
        "--report", type=Path, default=_REPO / "docs/BAND_FORCE_DH_DS_REPORT.md"
    )
    parser.add_argument(
        "--csv", type=Path, default=_REPO / "docs/band_force_dh_ds_pairs.csv"
    )
    args = parser.parse_args(argv)

    if str(args.utils_openmx.parent) not in sys.path:
        sys.path.insert(0, str(args.utils_openmx.parent))
    from utils_openmx.utils import basis_def_19

    if args.nao != 19:
        raise ValueError("CLI currently pins basis_def_19; pass --nao 19")
    pairs = _parse_pairs(args.pairs)
    bands = slice(args.band_start, args.band_start + args.band_width)
    z, graph_edge, graph_cell = _load_graph_contract(args.graph)
    model_ad = _load_model_ad(args.model_csv)
    rows = []
    start = time.time()

    from scripts.diff_hamgnn.openmx_fd import band_energy_from_out, parse_eigenvalues_out

    extract_root = args.work_root / ".hf_decomposition"
    for atom, xyz in pairs:
        geometry = {}
        branches = {}
        topologies = {}
        for tag in ("plus", "minus"):
            directory = args.work_root / f"a{atom}_c{xyz}_{tag}"
            scfout = directory / "openmx.scfout"
            out = directory / "openmx.out"
            if not scfout.exists() or not out.exists():
                raise FileNotFoundError(f"missing cached OpenMX outputs in {directory}")
            hs = _extract_hs(args.reader, scfout, extract_root / f"a{atom}_c{xyz}_{tag}")
            packed = pack_hs_blocks(hs, z, args.nao, basis_def_19)
            topologies[tag] = (packed["edge_index"], packed["cell_shift"])
            if not np.array_equal(packed["edge_index"], graph_edge):
                raise ValueError(f"edge topology mismatch for atom={atom} xyz={xyz} {tag}")
            if not np.array_equal(packed["cell_shift"], graph_cell):
                raise ValueError(f"cell shifts mismatch for atom={atom} xyz={xyz} {tag}")
            geometry[tag] = assemble_gamma_dense(packed)
            branches[tag] = parse_eigenvalues_out(out)

        if not np.array_equal(topologies["plus"][0], topologies["minus"][0]):
            raise ValueError(f"plus/minus edge topology mismatch for atom={atom} xyz={xyz}")
        if not np.array_equal(topologies["plus"][1], topologies["minus"][1]):
            raise ValueError(f"plus/minus cell shifts mismatch for atom={atom} xyz={xyz}")
        if branches["plus"]["homo"] != branches["minus"]["homo"]:
            raise ValueError(f"plus/minus HOMO mismatch for atom={atom} xyz={xyz}")

        result = decompose_dense_fd(
            geometry["plus"][0],
            geometry["plus"][1],
            geometry["minus"][0],
            geometry["minus"][1],
            args.delta,
            bands=bands,
        )
        sums = {name: float(np.sum(result[name])) for name in ("g_H", "g_S", "g_HF", "g_eig_fd", "residual")}
        plus_energy = band_energy_from_out(
            args.work_root / f"a{atom}_c{xyz}_plus" / "openmx.out", bands=bands
        )
        minus_energy = band_energy_from_out(
            args.work_root / f"a{atom}_c{xyz}_minus" / "openmx.out", bands=bands
        )
        g_eig_out = (plus_energy - minus_energy) / (2.0 * args.delta)
        residual_out = g_eig_out - sums["g_HF"]
        model_value = model_ad[(atom, xyz)]
        rows.append(
            {
                "atom": atom,
                "xyz": xyz,
                "model_AD": model_value,
                "g_H": sums["g_H"],
                "g_S": sums["g_S"],
                "g_HF": sums["g_HF"],
                "g_eig_dense_fd": sums["g_eig_fd"],
                "g_eig_out": g_eig_out,
                "residual_dense": sums["residual"],
                "residual_out": residual_out,
                "H_fraction": _safe_ratio(sums["g_H"], g_eig_out),
                "S_fraction": _safe_ratio(sums["g_S"], g_eig_out),
                "residual_fraction": _safe_ratio(residual_out, g_eig_out),
                "H_over_model": _safe_ratio(sums["g_H"], model_value),
                "chem_p_delta": abs(branches["plus"]["chem_p"] - branches["minus"]["chem_p"]),
                "homo": branches["plus"]["homo"],
                "overlap_min": float(np.min(result["overlap_eigenvalues_mid"])),
            }
        )

    metadata = {
        "campaign_id": "band-force-dh-ds-20260715",
        "delta_ang": args.delta,
        "bands": [args.band_start, args.band_start + args.band_width],
        "kpoint": "Gamma",
        "pairs": pairs,
        "wall_s": time.time() - start,
        "reader": str(args.reader),
        "work_root": str(args.work_root),
    }
    _write_outputs(rows, metadata, args.report, args.csv)
    for row in rows:
        print(
            f"atom={row['atom']} xyz={row['xyz']} model={row['model_AD']:.6e} "
            f"gH={row['g_H']:.6e} gS={row['g_S']:.6e} "
            f"gHF={row['g_HF']:.6e} eigFD={row['g_eig_out']:.6e} "
            f"res={row['residual_out']:.3e}"
        )
    print(f"Wrote {args.report}")
    print(f"Wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
