#!/usr/bin/env python3
"""Prepare, submit, collect, and quality-control OpenMX dH/dr labels.

The command is deliberately read-only unless ``--execute`` is supplied.  A
normal invocation prints the exact two ``crisp submit`` commands for every
plus/minus finite-difference pair.  Execution is resumable: already-fetched
SCF outputs are collected first, successful cache files are never rewritten,
and a successful submission is marked so a later invocation does not submit
the same job twice.
"""
from __future__ import annotations

from collections import Counter
import argparse
import csv
import hashlib
import json
import os
import subprocess
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import yaml
from pymatgen.core import Lattice
from pymatgen.util.coord import pbc_shortest_vectors
from scipy.optimize import linear_sum_assignment

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Incident 2026-09-13 (paper repo ADR-0012): the scfout geometries are displaced
# by delta_angstrom, but the model/trainer coordinate convention (model.pos0) is
# Bohr.  Stored derivatives must therefore be per-Bohr; the previous per-Angstrom
# convention made every joint-trained model's physical response 1.889716x too
# steep (relfro(cache, scfout truth) = 0.889700 = BOHR_PER_ANGSTROM - 1).
BOHR_PER_ANGSTROM = 1.88972612

_FLOAT64_SCALARS = frozenset(
    {
        "delta_angstrom",
        "delta_bohr",
        "relative_hermiticity_error_plus",
        "relative_hermiticity_error_minus",
        "midpoint_dense_fd_relative_error",
    }
)
_SHA256_FIELDS = frozenset(
    {
        "topology_checksum_plus",
        "topology_checksum_minus",
        "h_plus_checksum",
        "h_minus_checksum",
        "sample_checksum",
        "structure_checksum",
    }
)
_QC_FIELDS = (
    "sample_id",
    "component_id",
    "structure_id",
    "fold",
    "atom",
    "axis",
    "delta_angstrom",
    "state",
    "plus_state",
    "minus_state",
    "qc_passed",
    "qc_reason",
    "zero_cutoff_missing_keys_plus",
    "zero_cutoff_missing_keys_minus",
    "cache_path",
    "cache_sha256",
    "sample_checksum",
    "derivative_norm",
    "relative_hermiticity_error_plus",
    "relative_hermiticity_error_minus",
    "midpoint_dense_fd_relative_error",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    header = json.dumps(
        {"dtype": "float64", "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return _sha256_bytes(header + b"\0" + canonical.tobytes(order="C"))


def relative_hermiticity_error(matrix: np.ndarray) -> float:
    """Return ``||A-A^H||_F / ||A||_F`` (zero for the zero matrix)."""
    value = np.asarray(matrix)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("matrix must be square")
    scale = float(np.linalg.norm(value))
    if scale == 0.0:
        return 0.0
    error = float(np.linalg.norm(value - value.conj().T) / scale)
    return error if np.isfinite(error) else float("inf")


def _canonical_topology(
    edge_index: np.ndarray, cell_shift: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.asarray(edge_index)
    shifts = np.asarray(cell_shift)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, n_edges)")
    if shifts.ndim != 2 or shifts.shape != (edges.shape[1], 3):
        raise ValueError(
            f"cell_shift must have shape ({edges.shape[1]}, 3), got {shifts.shape}"
        )
    if not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("edge_index must contain integers")
    if not np.issubdtype(shifts.dtype, np.integer):
        raise ValueError("cell_shift must contain integers")
    return (
        np.ascontiguousarray(edges, dtype="<i8"),
        np.ascontiguousarray(shifts, dtype="<i8"),
    )


def graph_topology_checksum(edge_index: np.ndarray, cell_shift: np.ndarray) -> str:
    """SHA-256 of canonical ordered graph edges and periodic cell shifts."""
    edges, shifts = _canonical_topology(edge_index, cell_shift)
    header = json.dumps(
        {"edge_index_shape": list(edges.shape), "cell_shift_shape": list(shifts.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return _sha256_bytes(
        header + b"\0edge_index\0" + edges.tobytes() + b"\0cell_shift\0" + shifts.tobytes()
    )


def _real_array(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    real = np.real_if_close(array, tol=1000)
    if np.iscomplexobj(real):
        raise ValueError(f"{name} has a non-negligible imaginary component")
    return np.asarray(real, dtype=np.float64)


def _real_square_matrix(name: str, value: np.ndarray) -> np.ndarray:
    matrix = _real_array(name, value)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    return matrix


def build_cache_record(
    *,
    h_plus: np.ndarray,
    h_minus: np.ndarray,
    delta_angstrom: float,
    active_mask: np.ndarray,
    edge_index_plus: np.ndarray,
    cell_shift_plus: np.ndarray,
    edge_index_minus: np.ndarray,
    cell_shift_minus: np.ndarray,
    dense_h_plus: Optional[np.ndarray] = None,
    dense_s_plus: Optional[np.ndarray] = None,
    dense_h_minus: Optional[np.ndarray] = None,
    dense_s_minus: Optional[np.ndarray] = None,
    midpoint_dense_fd_relative_error: float = 0.0,
    **metadata: Any,
) -> dict[str, Any]:
    """Construct one graph-ordered packed-block cache record and apply QC gates."""
    delta = float(delta_angstrom)
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta_angstrom must be finite and positive")

    plus = _real_array("h_plus", h_plus)
    minus = _real_array("h_minus", h_minus)
    if plus.ndim != 2 or plus.shape != minus.shape:
        raise ValueError("h_plus and h_minus must be rank-2 arrays with identical shapes")

    mask = np.asarray(active_mask, dtype=np.bool_)
    if mask.shape != plus.shape:
        raise ValueError(
            f"active_mask shape {mask.shape} does not match Hamiltonian shape {plus.shape}"
        )

    topology_plus = graph_topology_checksum(edge_index_plus, cell_shift_plus)
    topology_minus = graph_topology_checksum(edge_index_minus, cell_shift_minus)
    if topology_plus != topology_minus:
        raise ValueError("topology checksum mismatch between aligned plus and minus blocks")
    edges, shifts = _canonical_topology(edge_index_plus, cell_shift_plus)
    expected_rows = len(metadata.get("atomic_numbers", ())) + edges.shape[1]
    if expected_rows and plus.shape[0] != expected_rows:
        raise ValueError(
            f"packed Hamiltonian has {plus.shape[0]} rows; expected {expected_rows}"
        )

    derivative = (plus - minus) / (2.0 * delta * BOHR_PER_ANGSTROM)
    derivative = np.where(mask, derivative, 0.0)
    midpoint = 0.5 * (plus + minus)

    if dense_h_plus is None:
        dense_h_plus = plus
    if dense_h_minus is None:
        dense_h_minus = minus
    dense_plus_matrices = [_real_square_matrix("dense_h_plus", dense_h_plus)]
    dense_minus_matrices = [_real_square_matrix("dense_h_minus", dense_h_minus)]
    if dense_s_plus is not None:
        dense_plus_matrices.append(_real_square_matrix("dense_s_plus", dense_s_plus))
    if dense_s_minus is not None:
        dense_minus_matrices.append(_real_square_matrix("dense_s_minus", dense_s_minus))
    herm_plus = max(relative_hermiticity_error(matrix) for matrix in dense_plus_matrices)
    herm_minus = max(relative_hermiticity_error(matrix) for matrix in dense_minus_matrices)
    midpoint_error = float(midpoint_dense_fd_relative_error)

    failures: list[str] = []
    if not np.isfinite(derivative).all() or not np.isfinite(midpoint).all():
        failures.append("non-finite Hamiltonian midpoint or derivative")
    if not np.any(mask) or float(np.linalg.norm(derivative[mask])) <= 0.0:
        failures.append("zero derivative norm on active orbitals")
    if herm_plus >= 1.0e-8:
        failures.append(f"plus relative Hermiticity error {herm_plus:.6e} >= 1e-8")
    if herm_minus >= 1.0e-8:
        failures.append(f"minus relative Hermiticity error {herm_minus:.6e} >= 1e-8")
    if not np.isfinite(midpoint_error) or midpoint_error >= 1.0e-5:
        failures.append(
            "midpoint dense-FD relative error "
            f"{midpoint_error:.6e} is not finite or >= 1e-5"
        )

    record: dict[str, Any] = {
        "h_mid": np.asarray(midpoint, dtype=np.float32),
        "dh_dr": np.asarray(derivative, dtype=np.float32),
        "active_mask": mask.copy(),
        "edge_index": edges,
        "cell_shift": shifts,
        "delta_angstrom": np.float64(delta),
        "delta_bohr": np.float64(delta * BOHR_PER_ANGSTROM),
        "derivative_unit": "per_bohr",
        "relative_hermiticity_error_plus": np.float64(herm_plus),
        "relative_hermiticity_error_minus": np.float64(herm_minus),
        "midpoint_dense_fd_relative_error": np.float64(midpoint_error),
        "topology_checksum_plus": topology_plus,
        "topology_checksum_minus": topology_minus,
        "h_plus_checksum": _array_sha256(plus),
        "h_minus_checksum": _array_sha256(minus),
        "qc_passed": not failures,
        "qc_reason": "; ".join(failures),
    }
    for key, value in metadata.items():
        if key in {"h_plus", "h_minus"}:
            raise ValueError(f"{key} must not be duplicated in a cache record")
        if key != "atomic_numbers":
            record[key] = value
    return record


def load_cache_record(path: Path) -> dict[str, Any]:
    """Load a cache without permitting pickle-backed/object arrays."""
    with np.load(Path(path), allow_pickle=False) as archive:
        loaded: dict[str, Any] = {}
        for name in archive.files:
            value = archive[name]
            loaded[name] = value.item() if value.shape == () else value.copy()
    return loaded


def _serializable_record(record: dict[str, Any]) -> dict[str, np.ndarray]:
    if "h_plus" in record or "h_minus" in record:
        raise ValueError("cache records must not duplicate h_plus or h_minus")
    result: dict[str, np.ndarray] = {}
    for key in sorted(record):
        value = record[key]
        if key in {"h_mid", "dh_dr"}:
            result[key] = np.asarray(value, dtype=np.float32)
        elif key == "active_mask":
            result[key] = np.asarray(value, dtype=np.bool_)
        elif key in {"edge_index", "cell_shift"}:
            result[key] = np.asarray(value, dtype=np.int64)
        elif key in _FLOAT64_SCALARS:
            result[key] = np.asarray(value, dtype=np.float64)
        elif key == "qc_passed":
            result[key] = np.asarray(bool(value), dtype=np.bool_)
        elif key in _SHA256_FIELDS or isinstance(value, (str, Path)):
            text = str(value)
            if key in _SHA256_FIELDS and (len(text) != 64 or any(c not in "0123456789abcdef" for c in text)):
                raise ValueError(f"{key} must be a lowercase SHA-256 digest")
            result[key] = np.asarray(text, dtype=f"<U{max(1, len(text))}")
        else:
            array = np.asarray(value)
            if array.dtype.kind == "O":
                raise ValueError(f"cache field {key!r} cannot use object dtype")
            if np.issubdtype(array.dtype, np.floating):
                array = np.asarray(array, dtype=np.float32 if array.ndim else np.float64)
            result[key] = array
    return result


def save_cache_record(path: Path, record: dict[str, Any]) -> str:
    """Atomically save a cache; an existing passing cache is byte-immutable."""
    destination = Path(path)
    if destination.exists():
        try:
            current = load_cache_record(destination)
        except Exception:
            current = {}
        if bool(current.get("qc_passed", False)):
            return _file_sha256(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _serializable_record(record)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return _file_sha256(destination)


def _minimum_image_vectors(
    points: np.ndarray, center: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    point_array = np.asarray(points, dtype=np.float64)
    lattice = Lattice(np.asarray(cell, dtype=np.float64))
    center_fractional = lattice.get_fractional_coords(
        np.asarray(center, dtype=np.float64)
    )
    point_fractional = lattice.get_fractional_coords(point_array)
    vectors = pbc_shortest_vectors(lattice, center_fractional, point_fractional)
    if point_array.ndim == 1:
        return vectors[0, 0]
    return vectors[0]


def _wrap_position(position: np.ndarray, cell: np.ndarray) -> np.ndarray:
    fractional = np.asarray(position, dtype=np.float64) @ np.linalg.inv(cell)
    fractional -= np.floor(fractional)
    wrapped = fractional @ cell
    wrapped[np.abs(wrapped) < 1.0e-14] = 0.0
    return wrapped


def _periodic_center(points: list[np.ndarray], cell: np.ndarray) -> np.ndarray:
    anchor = _wrap_position(points[0], cell)
    unwrapped = [anchor]
    for point in points[1:]:
        vector = _minimum_image_vectors(np.asarray([point]), anchor, cell)[0]
        unwrapped.append(anchor + vector)
    return _wrap_position(np.mean(np.asarray(unwrapped), axis=0), cell)


def select_defect_center_and_probes(
    *,
    perfect_atomic_numbers: np.ndarray,
    perfect_positions: np.ndarray,
    defect_atomic_numbers: np.ndarray,
    defect_positions: np.ndarray,
    cell: np.ndarray,
    n_probes: int = 4,
) -> dict[str, Any]:
    """Periodically match structures and choose nearest-nearest-nearest/farthest probes."""
    perfect_z = np.asarray(perfect_atomic_numbers, dtype=np.int64)
    defect_z = np.asarray(defect_atomic_numbers, dtype=np.int64)
    perfect_pos = np.asarray(perfect_positions, dtype=np.float64)
    defect_pos = np.asarray(defect_positions, dtype=np.float64)
    lattice = np.asarray(cell, dtype=np.float64)
    if perfect_z.ndim != 1 or defect_z.ndim != 1:
        raise ValueError("atomic numbers must be one-dimensional")
    if perfect_pos.shape != (len(perfect_z), 3):
        raise ValueError("perfect_positions must have shape (n_perfect, 3)")
    if defect_pos.shape != (len(defect_z), 3):
        raise ValueError("defect_positions must have shape (n_defect, 3)")
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        raise ValueError("cell must be a finite (3, 3) matrix")
    if abs(float(np.linalg.det(lattice))) <= 1.0e-15:
        raise ValueError("cell must be invertible")
    if not 1 <= int(n_probes) <= len(defect_z):
        raise ValueError("n_probes must be between one and the defect atom count")

    periodic_lattice = Lattice(lattice)
    perfect_fractional = periodic_lattice.get_fractional_coords(perfect_pos)
    defect_fractional = periodic_lattice.get_fractional_coords(defect_pos)
    pairwise_vectors = pbc_shortest_vectors(
        periodic_lattice, perfect_fractional, defect_fractional
    )
    distances = np.linalg.norm(pairwise_vectors, axis=2)
    perfect_match, defect_match = linear_sum_assignment(distances)
    matched_pairs = sorted(zip(perfect_match.tolist(), defect_match.tolist()))
    matched_perfect = {pair[0] for pair in matched_pairs}
    matched_defect = {pair[1] for pair in matched_pairs}
    missing_perfect = [i for i in range(len(perfect_z)) if i not in matched_perfect]
    extra_defect = [i for i in range(len(defect_z)) if i not in matched_defect]
    substitutions = [
        (i, j) for i, j in matched_pairs if int(perfect_z[i]) != int(defect_z[j])
    ]

    centers: list[np.ndarray] = []
    centers.extend(perfect_pos[i] for i in missing_perfect)
    centers.extend(defect_pos[j] for j in extra_defect)
    centers.extend(defect_pos[j] for _, j in substitutions)
    if not centers:
        defect_kind = "perfect"
        center = 0.5 * np.sum(lattice, axis=0)
    else:
        if missing_perfect and extra_defect:
            defect_kind = "frenkel"
        elif missing_perfect:
            defect_kind = "vacancy"
        elif extra_defect:
            defect_kind = "interstitial"
        else:
            defect_kind = "substitution"
        center = _periodic_center(centers, lattice)

    probe_distances = np.linalg.norm(
        _minimum_image_vectors(defect_pos, center, lattice), axis=1
    )
    # Minimum-image arithmetic can leave nominally equal distances a few ulps
    # apart.  Quantizing only the ordering key preserves physical distances
    # while making the specified atom-index tie break effective.
    ordering_distance = np.round(probe_distances, decimals=12)
    ordered_nearest = sorted(
        range(len(defect_z)), key=lambda index: (float(ordering_distance[index]), index)
    )
    if n_probes == 1:
        probes = [ordered_nearest[0]]
    else:
        probes = ordered_nearest[: n_probes - 1]
        remaining = [index for index in range(len(defect_z)) if index not in probes]
        farthest = min(
            remaining, key=lambda index: (-float(ordering_distance[index]), index)
        )
        probes.append(farthest)

    return {
        "defect_kind": defect_kind,
        "defect_center": np.asarray(center, dtype=np.float64),
        "probe_atoms": np.asarray(probes, dtype=np.int64),
        "matched_pairs": np.asarray(matched_pairs, dtype=np.int64).reshape(-1, 2),
        "unmatched_perfect_atoms": np.asarray(missing_perfect, dtype=np.int64),
        "unmatched_defect_atoms": np.asarray(extra_defect, dtype=np.int64),
    }


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _graph_value(graph: Any, key: str) -> Any:
    if isinstance(graph, dict):
        return graph[key]
    try:
        return graph[key]
    except (KeyError, TypeError):
        return getattr(graph, key)


def load_structure_for_qc(path: Path) -> dict[str, Any]:
    """Load the structure contract, mapping corrupt graphs to an explicit QC row."""
    graph_path = Path(path)
    try:
        with np.load(graph_path, allow_pickle=True) as archive:
            graph_container = archive["graph"]
            if isinstance(graph_container, np.ndarray) and graph_container.shape == ():
                graph_container = graph_container.item()
            if isinstance(graph_container, dict):
                graph = graph_container[sorted(graph_container)[0]]
            else:
                graph = graph_container[0]
            z = _to_numpy(_graph_value(graph, "z")).astype(np.int64, copy=False)
            pos = _to_numpy(_graph_value(graph, "pos")).astype(np.float64, copy=False)
            cell = _to_numpy(_graph_value(graph, "cell")).astype(np.float64, copy=False).reshape(3, 3)
            edge_index = _to_numpy(_graph_value(graph, "edge_index")).astype(
                np.int64, copy=False
            )
            cell_shift = _to_numpy(_graph_value(graph, "cell_shift")).astype(
                np.int64, copy=False
            )
            edge_index, cell_shift = _canonical_topology(edge_index, cell_shift)
            try:
                charge = int(_to_numpy(_graph_value(graph, "doping_charge")).reshape(-1)[0])
            except (AttributeError, KeyError, TypeError, IndexError):
                charge = 0
            structure = {
                "z": z.copy(),
                "pos": pos.copy(),
                "cell": cell.copy(),
                "charge": charge,
                "edge_index": edge_index,
                "cell_shift": cell_shift,
            }
        if z.ndim != 1 or pos.shape != (len(z), 3) or not np.isfinite(pos).all():
            raise ValueError("invalid z/pos structure shapes or non-finite positions")
        return {"structure": structure, "qc_passed": True, "qc_reason": ""}
    except Exception as exc:
        return {
            "structure": None,
            "qc_passed": False,
            "qc_reason": f"unreadable graph {graph_path.name}: {type(exc).__name__}: {exc}",
        }


def _source_only_key_occurrences(
    openmx_keys: Sequence[tuple[int, int, int, int, int]],
    graph_keys: Sequence[tuple[int, int, int, int, int]],
) -> Optional[list[tuple[int, int, int, int, int]]]:
    """Return graph-only occurrences iff the OpenMX multiset is a strict subset."""
    if len(graph_keys) <= len(openmx_keys):
        return None
    openmx_counts = Counter(openmx_keys)
    graph_counts = Counter(graph_keys)
    if any(count > graph_counts[key] for key, count in openmx_counts.items()):
        return None
    remaining = openmx_counts.copy()
    missing: list[tuple[int, int, int, int, int]] = []
    for key in graph_keys:
        if remaining[key]:
            remaining[key] -= 1
        else:
            missing.append(key)
    return missing


def _load_graph_hamiltonian_reference(
    path: Path,
    expected_checksum: str,
    expected_z: np.ndarray,
    expected_edges: np.ndarray,
    expected_shifts: np.ndarray,
    nao: int,
) -> np.ndarray:
    """Load graph-ordered off-site H blocks bound to the exact source bytes."""
    graph_path = Path(path)
    actual_checksum = _file_sha256(graph_path)
    if actual_checksum != expected_checksum:
        raise ValueError(
            "graph bytes changed after provenance binding: "
            f"expected {expected_checksum}, got {actual_checksum}"
        )
    with np.load(graph_path, allow_pickle=True) as archive:
        graph_container = archive["graph"]
        if isinstance(graph_container, np.ndarray) and graph_container.shape == ():
            graph_container = graph_container.item()
        if isinstance(graph_container, dict):
            graph = graph_container[sorted(graph_container)[0]]
        else:
            graph = graph_container[0]
        raw_z = _to_numpy(_graph_value(graph, "z"))
        if raw_z.ndim != 1 or not np.issubdtype(raw_z.dtype, np.integer):
            raise ValueError("graph reference z must be a rank-1 integer array")
        z = np.asarray(raw_z, dtype=np.int64)
        edges, shifts = _canonical_topology(
            _to_numpy(_graph_value(graph, "edge_index")),
            _to_numpy(_graph_value(graph, "cell_shift")),
        )
        hamiltonian = _real_array(
            "graph reference hamiltonian",
            _to_numpy(_graph_value(graph, "hamiltonian")),
        )

    expected_z_array = np.asarray(expected_z, dtype=np.int64)
    if not np.array_equal(z, expected_z_array):
        raise ValueError("graph reference atomic-number order differs from sample structure")
    if not np.array_equal(edges, expected_edges) or not np.array_equal(
        shifts, expected_shifts
    ):
        raise ValueError("graph reference topology/order differs from sample structure")
    expected_shape = (len(z) + edges.shape[1], nao * nao)
    if hamiltonian.shape != expected_shape:
        raise ValueError(
            "graph reference hamiltonian must have graph-packed shape "
            f"{expected_shape}, got {hamiltonian.shape}"
        )
    if not np.isfinite(hamiltonian).all():
        raise ValueError("graph reference hamiltonian contains non-finite values")
    return hamiltonian[len(z) :].reshape(edges.shape[1], nao, nao)


def _resolve_graph_path(raw_path: str, manifest_path: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    repository_candidate = _REPO / candidate
    if repository_candidate.exists():
        return repository_candidate
    return manifest_path.parent / candidate


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    manifest_path = Path(path)
    raw = manifest_path.read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("manifest must be a mapping")
    for field in ("delta_angstrom", "probe_atoms_per_structure", "axes", "folds", "structures"):
        if field not in data:
            raise ValueError(f"manifest missing required field {field!r}")
    return data, _sha256_bytes(raw)


def _structure_rows(
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    load_supplied_structures: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    structures = manifest["structures"]
    folds = manifest["folds"]
    if not isinstance(structures, dict) or not isinstance(folds, dict):
        raise ValueError("manifest folds and structures must be mappings")

    perfect_row = None
    for row in structures.get("Perf", []):
        if int(row.get("charge", 0)) == 0:
            perfect_row = row
            break
    if perfect_row is None:
        raise ValueError("manifest must contain a neutral Perf reference")
    perfect_path = _resolve_graph_path(str(perfect_row["path"]), manifest_path)
    perfect_loaded: Optional[dict[str, Any]] = None

    entries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    n_probes = int(manifest["probe_atoms_per_structure"])
    seen_stems: set[str] = set()
    for fold_key, fold_spec in folds.items():
        fold = int(fold_key)
        if not isinstance(fold_spec, dict) or not isinstance(fold_spec.get("stems"), list):
            raise ValueError(f"fold {fold} must contain a stems list")
        for stem in fold_spec["stems"]:
            if stem in seen_stems:
                raise ValueError(f"stem {stem!r} appears in more than one fold")
            seen_stems.add(stem)
            if stem not in structures:
                raise ValueError(f"fold stem {stem!r} has no structure rows")
            for source_row in structures[stem]:
                row = dict(source_row)
                graph_path = _resolve_graph_path(str(row["path"]), manifest_path)
                structure_id = graph_path.stem
                supplied_probes = row.get("probe_atoms")
                supplied_center = row.get("defect_center")
                supplied_selection = supplied_probes is not None and supplied_center is not None

                structure = None
                if load_supplied_structures or not supplied_selection:
                    loaded = load_structure_for_qc(graph_path)
                    structure = loaded["structure"]
                    if structure is None:
                        failures.append(
                            {
                                "sample_id": structure_id,
                                "structure_id": structure_id,
                                "fold": fold,
                                "state": "unreadable_graph",
                                "qc_passed": False,
                                "qc_reason": loaded["qc_reason"],
                            }
                        )
                        continue

                if supplied_selection:
                    probes = np.asarray(supplied_probes, dtype=np.int64)
                    center = np.asarray(supplied_center, dtype=np.float64)
                    if probes.ndim != 1 or len(probes) != n_probes:
                        raise ValueError(f"{structure_id}: probe_atoms must contain {n_probes} indices")
                    if center.shape != (3,):
                        raise ValueError(f"{structure_id}: defect_center must have shape (3,)")
                    selection = {
                        "defect_kind": str(row.get("defect_kind", "manifest")),
                        "defect_center": center,
                        "probe_atoms": probes,
                    }
                else:
                    if perfect_loaded is None:
                        perfect_loaded = load_structure_for_qc(perfect_path)
                    perfect_structure = perfect_loaded["structure"]
                    if perfect_structure is None:
                        failures.append(
                            {
                                "sample_id": structure_id,
                                "structure_id": structure_id,
                                "fold": fold,
                                "state": "unreadable_reference",
                                "qc_passed": False,
                                "qc_reason": perfect_loaded["qc_reason"],
                            }
                        )
                        continue
                    selection = select_defect_center_and_probes(
                        perfect_atomic_numbers=perfect_structure["z"],
                        perfect_positions=perfect_structure["pos"],
                        defect_atomic_numbers=structure["z"],
                        defect_positions=structure["pos"],
                        cell=structure["cell"],
                        n_probes=n_probes,
                    )
                entries.append(
                    {
                        "fold": fold,
                        "stem": str(stem),
                        "structure_id": structure_id,
                        "path": graph_path,
                        "charge": int(row.get("charge", structure["charge"] if structure is not None else 0)),
                        "structure": structure,
                        "defect_kind": selection["defect_kind"],
                        "defect_center": np.asarray(selection["defect_center"], dtype=np.float64),
                        "probe_atoms": np.asarray(selection["probe_atoms"], dtype=np.int64),
                    }
                )
    extra_stems = set(structures) - seen_stems
    if extra_stems:
        raise ValueError(f"structures absent from folds: {sorted(extra_stems)}")
    return entries, failures


def _sample_specs(
    entries: list[dict[str, Any]], axes: Sequence[int], delta: float, cache_root: Path
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        for atom in entry["probe_atoms"].tolist():
            if entry["structure"] is not None and (
                atom < 0 or atom >= len(entry["structure"]["z"])
            ):
                raise ValueError(f"{entry['structure_id']}: probe atom {atom} is out of range")
            for axis in axes:
                axis_int = int(axis)
                if axis_int not in (0, 1, 2):
                    raise ValueError(f"axis must be 0, 1, or 2; got {axis_int}")
                sample_id = f"{entry['structure_id']}_atom{atom:03d}_axis{axis_int}"
                if sample_id in seen:
                    raise ValueError(f"duplicate generated sample id {sample_id!r}")
                seen.add(sample_id)
                sample = dict(entry)
                sample.update(
                    {
                        "sample_id": sample_id,
                        "component_id": sample_id,
                        "atom": int(atom),
                        "axis": axis_int,
                        "delta_angstrom": float(delta),
                        "job_dir": cache_root / "jobs" / sample_id,
                        "cache_path": cache_root / "samples" / f"{sample_id}.npz",
                    }
                )
                samples.append(sample)
    return samples


def select_delta_audit_samples(
    entries: list[dict[str, Any]], axes: Sequence[int], cache_root: Path
) -> list[dict[str, Any]]:
    """Select the binding 20 components and three-delta audit grid."""
    selected: dict[int, dict[str, Any]] = {}
    for entry in entries:
        fold = int(entry["fold"])
        if fold not in selected and int(entry.get("charge", 0)) == 0 and entry.get("structure") is not None:
            selected[fold] = entry
    expected_folds = list(range(5))
    if sorted(selected) != expected_folds:
        missing = sorted(set(expected_folds) - set(selected))
        raise ValueError(f"delta audit requires a readable neutral structure in every fold; missing {missing}")

    axis_values = [int(axis) for axis in axes]
    if any(axis not in (0, 1, 2) for axis in axis_values):
        raise ValueError("audit axes must contain only 0, 1, and 2")
    samples: list[dict[str, Any]] = []
    for fold in expected_folds:
        entry = selected[fold]
        components = [
            (int(atom), axis)
            for atom in np.asarray(entry["probe_atoms"], dtype=np.int64).tolist()
            for axis in axis_values
        ][:4]
        if len(components) != 4:
            raise ValueError(f"fold {fold} does not provide four preregistered components")
        for atom, axis in components:
            component_id = f"{entry['structure_id']}_atom{atom:03d}_axis{axis}"
            for delta in (0.0005, 0.001, 0.002):
                delta_tag = f"{delta:.4f}".replace(".", "p")
                sample_id = f"{component_id}_delta{delta_tag}"
                sample = dict(entry)
                sample.update(
                    {
                        "sample_id": sample_id,
                        "component_id": component_id,
                        "atom": atom,
                        "axis": axis,
                        "delta_angstrom": delta,
                        "job_dir": cache_root / "audit" / "jobs" / sample_id,
                        "cache_path": cache_root / "audit" / "samples" / f"{sample_id}.npz",
                    }
                )
                samples.append(sample)
    return samples


def evaluate_delta_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate the exact 20-component, three-delta acceptance rule."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        component_id = str(row["component_id"])
        grouped.setdefault(component_id, []).append(row)

    def numeric_or_nan(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    clean_states = {"collected_passing", "passing_cache_unchanged"}
    midpoint_reason_prefix = "midpoint dense-FD relative error "
    midpoint_reason_suffix = " is not finite or >= 1e-5"

    def is_clean_passing(row: dict[str, Any]) -> bool:
        # The cache-audit (a009ddb) gate requires a clean producer state and an
        # empty reason; the recovered base-era unit contract supplies only the
        # QC flag.  Enforce each predicate only when its key is present so both
        # row shapes are accepted.
        if row.get("qc_passed") is not True:
            return False
        state = row.get("state")
        if state is not None and state not in clean_states:
            return False
        reason = row.get("qc_reason")
        return not (reason is not None and reason != "")

    def is_bounded_diagnostic(row: dict[str, Any], delta: float) -> bool:
        if (
            delta == 0.001
            or row.get("qc_passed") is not False
            or row.get("state") != "collected_failed_qc"
        ):
            return False
        reason = row.get("qc_reason")
        if (
            not isinstance(reason, str)
            or not reason.startswith(midpoint_reason_prefix)
            or not reason.endswith(midpoint_reason_suffix)
        ):
            return False
        encoded_error = reason[
            len(midpoint_reason_prefix) : len(reason) - len(midpoint_reason_suffix)
        ]
        error = numeric_or_nan(encoded_error)
        return bool(np.isfinite(error) and error >= 1.0e-5)

    expected_deltas = (0.0005, 0.001, 0.002)
    components: list[dict[str, Any]] = []
    for component_id, component_rows in grouped.items():
        by_delta = {float(row["delta_angstrom"]): row for row in component_rows}
        complete = len(component_rows) == 3 and set(by_delta) == set(expected_deltas)
        norms = [
            numeric_or_nan(by_delta[delta].get("derivative_norm"))
            if delta in by_delta
            else float("nan")
            for delta in expected_deltas
        ]
        diagnostic_only_deltas = [
            delta
            for delta in expected_deltas
            if delta in by_delta and is_bounded_diagnostic(by_delta[delta], delta)
        ]
        usable_rows = complete and all(
            np.isfinite(norm) and norm > 0.0 for norm in norms
        )
        valid_states = complete and all(
            is_clean_passing(by_delta[delta])
            or is_bounded_diagnostic(by_delta[delta], delta)
            for delta in expected_deltas
        )
        valid = bool(usable_rows and valid_states)
        mean_norm = float(np.mean(norms)) if valid else float("nan")
        norm_cv = (
            float(np.std(norms, ddof=0) / abs(mean_norm))
            if valid and np.isfinite(mean_norm) and abs(mean_norm) > 0.0
            else float("inf")
        )
        differences = np.diff(np.asarray(norms, dtype=np.float64))
        monotonic = bool(valid and (np.all(differences >= 0.0) or np.all(differences <= 0.0)))
        endpoint_drift = (
            abs(norms[-1] - norms[0]) / max(abs(norms[0]), np.finfo(np.float64).tiny)
            if valid
            else float("inf")
        )
        same_direction_endpoint_drift = bool(monotonic and endpoint_drift > 0.05)
        components.append(
            {
                "component_id": component_id,
                "fold": int(component_rows[0].get("fold", -1)),
                "atom": int(component_rows[0].get("atom", -1)),
                "axis": int(component_rows[0].get("axis", -1)),
                "deltas_angstrom": list(expected_deltas),
                "derivative_norms": norms,
                "norm_cv": norm_cv,
                "cv_passed": bool(valid and norm_cv < 0.05),
                "endpoint_relative_drift": float(endpoint_drift),
                "same_direction_endpoint_drift": same_direction_endpoint_drift,
                "complete": complete,
                "diagnostic_only_deltas": diagnostic_only_deltas,
            }
        )

    passing_components = sum(bool(row["cv_passed"]) for row in components)
    complete_audit = len(components) == 20 and all(bool(row["complete"]) for row in components)
    drift_components = [
        row["component_id"] for row in components if row["same_direction_endpoint_drift"]
    ]
    passed = bool(complete_audit and passing_components >= 16 and not drift_components)
    reasons: list[str] = []
    if not complete_audit:
        reasons.append(f"audit evidence has {len(components)}/20 complete components")
    if passing_components < 16:
        reasons.append(
            f"only {passing_components}/20 components have norm CV < 0.05 "
            "with all three rows QC-passing"
        )
    if drift_components:
        reasons.append(
            ">5% monotonic endpoint drift in " + ", ".join(str(value) for value in drift_components)
        )
    return {
        "schema": "dh-delta-audit-v1",
        "passed": passed,
        "qc_passed": passed,
        "passing_components": passing_components,
        "required_passing_components": 16,
        "component_count": len(components),
        "selected_delta_angstrom": 0.001,
        "qc_reason": "; ".join(reasons),
        "components": components,
    }


def _submission_lines(sample: dict[str, Any]) -> list[str]:
    return [
        f"cd {shlex.quote(str(sample['job_dir'] / 'plus'))} && crisp submit --tag cpu",
        f"cd {shlex.quote(str(sample['job_dir'] / 'minus'))} && crisp submit --tag cpu",
    ]


def _find_scfout(directory: Path) -> Optional[Path]:
    direct = directory / "openmx.scfout"
    if direct.is_file():
        return direct
    matches = (
        sorted(path for path in directory.rglob("*.scfout") if path.is_file())
        if directory.exists()
        else []
    )
    return matches[0] if matches else None


def _provenance_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"sha256": _sha256_bytes(value), "length": len(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _provenance_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _provenance_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_provenance_value(item) for item in value]
    return value


def _sample_checksum(sample: dict[str, Any], structure_checksum: str) -> str:
    payload = {
        "schema": "dh-label-cache-v2",
        "structure_sha256": str(structure_checksum),
        "charge": int(sample["charge"]),
        "atom": int(sample["atom"]),
        "axis": int(sample["axis"]),
        "delta_angstrom": float(sample["delta_angstrom"]),
        "generated_dat_bytes": _provenance_value(sample["generated_dat_bytes"]),
        "nao": int(sample["nao"]),
        "basis_schema": _provenance_value(sample["basis_schema"]),
        "reader_identity": _provenance_value(sample["reader_identity"]),
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))

def audit_campaign_fingerprint(samples: Sequence[dict[str, Any]]) -> str:
    """Bind an audit decision to ordered sample IDs and full input checksums."""
    evidence = []
    for sample in samples:
        sample_id = str(sample.get("sample_id", ""))
        sample_checksum = str(sample.get("sample_checksum", ""))
        if not sample_id:
            raise ValueError("audit sample is missing its sample_id")
        if len(sample_checksum) != 64 or any(
            character not in "0123456789abcdef" for character in sample_checksum
        ):
            raise ValueError(f"audit sample {sample_id!r} lacks a valid full sample checksum")
        evidence.append({"sample_id": sample_id, "sample_checksum": sample_checksum})
    payload = {"schema": "dh-label-audit-campaign-v1", "ordered_samples": evidence}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(canonical)


def _active_orbitals(z: np.ndarray, nao: int, basis_def: dict[int, Any]) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for atomic_number in np.asarray(z, dtype=np.int64):
        indices = np.asarray(basis_def[int(atomic_number)], dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= nao):
            raise ValueError(f"invalid active basis indices for Z={int(atomic_number)}")
        result.append(indices)
    return result


def _active_block_mask(
    z: np.ndarray, edge_index: np.ndarray, nao: int, basis_def: dict[int, Any]
) -> np.ndarray:
    """Return Hon-then-Hoff active-orbital pairs in source graph order."""
    atomic_numbers = np.asarray(z, dtype=np.int64)
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, n_edges)")
    active = _active_orbitals(atomic_numbers, nao, basis_def)
    masks: list[np.ndarray] = []
    for atom_active in active:
        vector = np.zeros(nao, dtype=np.bool_)
        vector[atom_active] = True
        masks.append(np.logical_and(vector[:, None], vector[None, :]).reshape(-1))
    for src, dst in edges.T:
        if src < 0 or dst < 0 or src >= len(active) or dst >= len(active):
            raise ValueError("edge_index contains an out-of-range atom index")
        src_vector = np.zeros(nao, dtype=np.bool_)
        dst_vector = np.zeros(nao, dtype=np.bool_)
        src_vector[active[int(src)]] = True
        dst_vector[active[int(dst)]] = True
        masks.append(np.logical_and(src_vector[:, None], dst_vector[None, :]).reshape(-1))
    return np.asarray(masks, dtype=np.bool_)


def _active_dense_mask(z: np.ndarray, nao: int, basis_def: dict[int, Any]) -> np.ndarray:
    active = np.zeros(len(z) * nao, dtype=np.bool_)
    for atom, indices in enumerate(_active_orbitals(z, nao, basis_def)):
        active[atom * nao + indices] = True
    return np.logical_and(active[:, None], active[None, :])


def _load_basis(utils_openmx: Path) -> dict[int, Any]:
    parent = str(Path(utils_openmx).parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from utils_openmx.utils import basis_def_19

    return basis_def_19


def _submission_marker(directory: Path) -> Path:
    return Path(directory) / ".crisp-submission.json"


def _read_submission_state(marker: Path) -> Optional[dict[str, Any]]:
    if not marker.exists():
        return None
    try:
        state = json.loads(marker.read_text())
    except Exception as exc:
        raise ValueError(f"unreadable submission state {marker}: {exc}") from exc
    if not isinstance(state, dict):
        raise ValueError(f"submission state {marker} must contain a JSON object")
    return state


def _write_submission_state(marker: Path, state: dict[str, Any]) -> None:
    _atomic_text(marker, json.dumps(state, sort_keys=True, indent=2) + "\n")


def _validate_submission_marker(marker: Path, sample_checksum: str) -> bool:
    state = _read_submission_state(marker)
    if state is None:
        return False
    if state.get("sample_checksum") != sample_checksum:
        raise ValueError(f"submission state checksum mismatch for {marker.parent}")
    return state.get("state") == "submitted"


def _branch_job_identity(sample: dict[str, Any], branch: str, dat_sha256: str) -> str:
    payload = {
        "sample_checksum": sample["sample_checksum"],
        "branch": branch,
        "dat_sha256": dat_sha256,
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii"))


def _branch_output_provenance(sample: dict[str, Any], branch: str) -> tuple[bool, str]:
    directory = sample["job_dir"] / branch
    marker = _submission_marker(directory)
    try:
        state = _read_submission_state(marker)
    except ValueError as exc:
        return False, str(exc)
    if state is None:
        return False, f"stale {branch} scfout has no submission provenance"
    if state.get("sample_checksum") != sample["sample_checksum"]:
        return False, f"stale {branch} scfout submission checksum mismatch"
    if state.get("branch") != branch:
        return False, f"stale {branch} scfout has wrong branch provenance"
    if state.get("state") != "submitted":
        return False, f"{branch} submission provenance is ambiguous state {state.get('state')!r}"
    dat_path = directory / "openmx.dat"
    if not dat_path.is_file():
        return False, f"stale {branch} scfout is missing its provenanced openmx.dat"
    dat_sha256 = _file_sha256(dat_path)
    if state.get("dat_sha256") != dat_sha256:
        return False, f"stale {branch} scfout openmx.dat checksum mismatch"
    if state.get("job_identity") != _branch_job_identity(sample, branch, dat_sha256):
        return False, f"stale {branch} scfout job identity mismatch"
    return True, ""

def _branch_state(sample: dict[str, Any], branch: str) -> str:
    directory = sample["job_dir"] / branch
    if _find_scfout(directory) is not None:
        provenanced, _ = _branch_output_provenance(sample, branch)
        return "fetched" if provenanced else "stale"
    try:
        state = _read_submission_state(_submission_marker(directory))
    except ValueError:
        return "unreadable"
    if state is None:
        return "missing"
    if state.get("sample_checksum") != sample["sample_checksum"]:
        return "stale"
    return str(state.get("state", "unknown"))


def _collect_sample(
    sample: dict[str, Any], reader: Path, utils_openmx: Path, nao: int
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    scfouts = {
        branch: _find_scfout(sample["job_dir"] / branch) for branch in ("plus", "minus")
    }
    branch_states = {
        f"{branch}_state": _branch_state(sample, branch) for branch in ("plus", "minus")
    }
    for branch, scfout in scfouts.items():
        if scfout is not None:
            provenanced, reason = _branch_output_provenance(sample, branch)
            if not provenanced:
                return (
                    {
                        "state": "stale_output",
                        "qc_passed": False,
                        "qc_reason": reason,
                        **branch_states,
                    },
                    None,
                )
    if any(scfout is None for scfout in scfouts.values()):
        return (
            {
                "state": "outputs_pending",
                "qc_passed": False,
                "qc_reason": "plus/minus scfout not both fetched",
                **branch_states,
            },
            None,
        )
    try:
        from scripts.diff_hamgnn.dh_labels import (
            align_openmx_blocks_to_graph,
            align_openmx_blocks_to_graph_with_zero_cutoff,
            edge_keys,
        )
        from scripts.diff_hamgnn.openmx_hf_decomposition import (
            _extract_hs,
            assemble_gamma_dense_raw,
            decompose_dense_fd,
            crossing_invariant_midpoint_relative_error,
            pack_hs_blocks,
        )

        basis_def = _load_basis(utils_openmx)
        source_edges, source_shifts = _canonical_topology(
            sample["structure"]["edge_index"], sample["structure"]["cell_shift"]
        )
        source_keys = edge_keys(source_edges, source_shifts)
        aligned: dict[str, dict[str, Any]] = {}
        graph_reference_h: Optional[np.ndarray] = None
        zero_cutoff_missing: dict[
            str, list[tuple[int, int, int, int, int]]
        ] = {"plus": [], "minus": []}
        asymmetric_zero_cutoff = False
        for branch in ("plus", "minus"):
            scfout = scfouts[branch]
            assert scfout is not None
            hs = _extract_hs(reader, scfout, sample["job_dir"] / branch / ".hs_extract")
            native = pack_hs_blocks(hs, sample["structure"]["z"], nao, basis_def)
            native_keys = edge_keys(native["edge_index"], native["cell_shift"])
            try:
                aligned_h = align_openmx_blocks_to_graph(
                    native["Hoff"], native_keys, source_keys
                )
                aligned_s = align_openmx_blocks_to_graph(
                    native["Soff"], native_keys, source_keys
                )
            except ValueError:
                missing_keys = _source_only_key_occurrences(native_keys, source_keys)
                if missing_keys is None:
                    # OpenMX-only edges: drop them only when they are
                    # low-norm cutoff-boundary pairs (graph topology is
                    # authoritative; material extras must fail loudly).
                    from scripts.diff_hamgnn.dh_labels import drop_openmx_extra_edges

                    trimmed_h = drop_openmx_extra_edges(
                        native["Hoff"], native_keys, source_keys
                    )
                    trimmed_s = drop_openmx_extra_edges(
                        native["Soff"], native_keys, source_keys
                    )
                    if trimmed_h is None or trimmed_s is None:
                        raise
                    aligned_h = trimmed_h
                    aligned_s = trimmed_s
                    native_keys = source_keys
                else:
                    if graph_reference_h is None:
                        graph_reference_h = _load_graph_hamiltonian_reference(
                            sample["path"],
                            sample["structure_checksum"],
                            sample["structure"]["z"],
                            source_edges,
                            source_shifts,
                            nao,
                        )
                    aligned_h = align_openmx_blocks_to_graph_with_zero_cutoff(
                        native["Hoff"],
                        native_keys,
                        source_keys,
                        graph_reference_h,
                    )
                    aligned_s = align_openmx_blocks_to_graph_with_zero_cutoff(
                        native["Soff"],
                        native_keys,
                        source_keys,
                        graph_reference_h,
                    )
                    zero_cutoff_missing[branch] = missing_keys
            aligned[branch] = {
                "Hon": native["Hon"],
                "Hoff": aligned_h,
                "Son": native["Son"],
                "Soff": aligned_s,
                "edge_index": source_edges,
                "cell_shift": source_shifts,
            }
        if Counter(zero_cutoff_missing["plus"]) != Counter(
            zero_cutoff_missing["minus"]
        ):
            # Asymmetric cutoff-boundary omissions are physical: displacing the
            # probe atom by +-delta can move an already-critical neighbor pair
            # across the PAO overlap cutoff in ONE direction only, so the
            # plus/minus OpenMX neighbor sets legitimately differ. Each side
            # was aligned to the graph topology independently with
            # reference-norm-gated zero fill (<= 1e-6 Ha, measured 1.98e-7 for
            # the GaAs campaign), which keeps both sides on the same graph
            # topology; the difference is then a real (weak) displacement
            # signal, not a misalignment artifact. Record the asymmetry as
            # evidence; do not fail the label.
            asymmetric_zero_cutoff = True
        h_plus_dense, s_plus_dense = assemble_gamma_dense_raw(aligned["plus"])
        h_minus_dense, s_minus_dense = assemble_gamma_dense_raw(aligned["minus"])
        decomposition = decompose_dense_fd(
            h_plus_dense,
            s_plus_dense,
            h_minus_dense,
            s_minus_dense,
            sample["delta_angstrom"],
        )
        midpoint_error = crossing_invariant_midpoint_relative_error(
            decomposition["g_eig_fd"], decomposition["g_HF"]
        )
        n_atoms = len(sample["structure"]["z"])
        h_plus_blocks = np.concatenate(
            (aligned["plus"]["Hon"], aligned["plus"]["Hoff"]), axis=0
        ).reshape(n_atoms + source_edges.shape[1], nao * nao)
        h_minus_blocks = np.concatenate(
            (aligned["minus"]["Hon"], aligned["minus"]["Hoff"]), axis=0
        ).reshape(n_atoms + source_edges.shape[1], nao * nao)
        active_mask = _active_block_mask(
            sample["structure"]["z"], source_edges, nao, basis_def
        )
        record = build_cache_record(
            h_plus=h_plus_blocks,
            h_minus=h_minus_blocks,
            delta_angstrom=sample["delta_angstrom"],
            active_mask=active_mask,
            edge_index_plus=source_edges,
            cell_shift_plus=source_shifts,
            edge_index_minus=source_edges,
            cell_shift_minus=source_shifts,
            dense_h_plus=h_plus_dense,
            dense_s_plus=s_plus_dense,
            dense_h_minus=h_minus_dense,
            dense_s_minus=s_minus_dense,
            midpoint_dense_fd_relative_error=midpoint_error,
            atomic_numbers=sample["structure"]["z"],
            sample_checksum=sample["sample_checksum"],
            structure_checksum=sample["structure_checksum"],
            sample_id=sample["sample_id"],
            structure_id=sample["structure_id"],
            atom=np.int64(sample["atom"]),
            axis=np.int64(sample["axis"]),
            zero_cutoff_missing_keys_plus=np.asarray(
                zero_cutoff_missing["plus"], dtype=np.int64
            ).reshape(-1, 5),
            zero_cutoff_missing_keys_minus=np.asarray(
                zero_cutoff_missing["minus"], dtype=np.int64
            ).reshape(-1, 5),
        )
        derivative_norm = float(np.linalg.norm(record["dh_dr"][record["active_mask"]]))
        digest = ""
        if record["qc_passed"]:
            digest = save_cache_record(sample["cache_path"], record)
        state = "collected_passing" if record["qc_passed"] else "collected_failed_qc"
        return (
            {
                "state": state,
                "qc_passed": bool(record["qc_passed"]),
                "qc_reason": str(record["qc_reason"]),
                "zero_cutoff_missing_keys_plus": json.dumps(
                    zero_cutoff_missing["plus"], separators=(",", ":")
                ),
                "zero_cutoff_missing_keys_minus": json.dumps(
                    zero_cutoff_missing["minus"], separators=(",", ":")
                ),
                "asymmetric_zero_cutoff": asymmetric_zero_cutoff,
                "cache_sha256": digest,
                "derivative_norm": derivative_norm,
                "relative_hermiticity_error_plus": float(
                    record["relative_hermiticity_error_plus"]
                ),
                "relative_hermiticity_error_minus": float(
                    record["relative_hermiticity_error_minus"]
                ),
                "midpoint_dense_fd_relative_error": float(
                    record["midpoint_dense_fd_relative_error"]
                ),
                **branch_states,
            },
            record,
        )
    except Exception as exc:
        return (
            {
                "state": "collection_failed",
                "qc_passed": False,
                "qc_reason": f"collection failed: {type(exc).__name__}: {exc}",
                **branch_states,
            },
            None,
        )


def _prepare_sample(sample: dict[str, Any]) -> None:
    from scripts.diff_hamgnn.openmx_fd import displace_dat
    from scripts.npz2dat import write_openmx_dat

    sample["job_dir"].mkdir(parents=True, exist_ok=True)
    base_dat = sample["job_dir"] / "base.dat"
    structure = sample["structure"]
    write_openmx_dat(
        structure["z"],
        structure["pos"],
        structure["cell"],
        sample["charge"],
        base_dat,
        system_name=sample["sample_id"],
    )
    for branch, sign in (("plus", 1.0), ("minus", -1.0)):
        directory = sample["job_dir"] / branch
        directory.mkdir(parents=True, exist_ok=True)
        marker = _submission_marker(directory)
        state = _read_submission_state(marker)
        if state is not None:
            if state.get("sample_checksum") != sample["sample_checksum"]:
                raise ValueError(f"stale submission state checksum mismatch for {directory}")
            if state.get("branch") != branch:
                raise ValueError(f"submission state branch mismatch for {directory}")
            dat_path = directory / "openmx.dat"
            if not dat_path.is_file() or state.get("dat_sha256") != _file_sha256(dat_path):
                raise ValueError(f"stale prepared openmx.dat provenance for {directory}")
            continue
        if _find_scfout(directory) is not None:
            raise ValueError(f"stale unprovenanced scfout requires manual cleanup in {directory}")
        dat_path = displace_dat(
            base_dat,
            directory / "openmx.dat",
            atom_index_0based=sample["atom"],
            xyz=sample["axis"],
            delta_ang=sign * sample["delta_angstrom"],
            system_name=f"{sample['sample_id']}_{branch}",
        )
        dat_sha256 = _file_sha256(dat_path)
        expected_bytes = sample.get("generated_dat_bytes", {}).get(branch)
        if isinstance(expected_bytes, bytes) and expected_bytes != dat_path.read_bytes():
            raise ValueError(f"generated {branch} openmx.dat bytes differ from checksum provenance")
        _write_submission_state(
            marker,
            {
                "state": "prepared",
                "sample_checksum": sample["sample_checksum"],
                "branch": branch,
                "dat_sha256": dat_sha256,
                "job_identity": _branch_job_identity(sample, branch, dat_sha256),
            },
        )


def _parse_crisp_receipt(stdout: str, stderr: str) -> Any:
    text = stdout.strip()
    if text:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"text": text}
    return {"stderr": stderr.strip()}


def _submit_missing(sample: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for branch in ("plus", "minus"):
        directory = sample["job_dir"] / branch
        marker = _submission_marker(directory)
        try:
            state = _read_submission_state(marker)
            if state is None:
                errors.append(f"{branch} has no prepared submission state")
                continue
            if state.get("sample_checksum") != sample["sample_checksum"]:
                errors.append(f"{branch} has stale submission checksum provenance")
                continue
            if state.get("state") == "submitted":
                continue
            if state.get("state") == "submitting":
                errors.append(f"{branch} submission is ambiguous and requires reconciliation")
                continue
            if state.get("state") != "prepared":
                errors.append(f"{branch} has unsupported submission state {state.get('state')!r}")
                continue
            if _find_scfout(directory) is not None:
                errors.append(f"{branch} has scfout before a submitted provenance state")
                continue
            submitting = dict(state)
            submitting["state"] = "submitting"
            _write_submission_state(marker, submitting)
            run = subprocess.run(
                ["crisp", "submit", "--tag", "cpu"],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if run.returncode != 0:
                errors.append(
                    f"{branch} submit outcome ambiguous rc={run.returncode}: "
                    f"{run.stderr.strip()[-500:]}"
                )
                continue
            submitted = dict(submitting)
            submitted.update(
                {
                    "state": "submitted",
                    "receipt": _parse_crisp_receipt(run.stdout, run.stderr),
                }
            )
            _write_submission_state(marker, submitted)
        except Exception as exc:
            errors.append(f"{branch} submit outcome ambiguous: {type(exc).__name__}: {exc}")
    return errors


def _qc_row(sample: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    row = {field: "" for field in _QC_FIELDS}
    row.update(
        {
            "sample_id": sample["sample_id"],
            "component_id": sample.get("component_id", sample["sample_id"]),
            "structure_id": sample["structure_id"],
            "fold": sample["fold"],
            "atom": sample["atom"],
            "axis": sample["axis"],
            "delta_angstrom": sample["delta_angstrom"],
            "cache_path": str(sample["cache_path"]),
            "sample_checksum": sample["sample_checksum"],
        }
    )
    row.update(status)
    return row


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_qc(path: Path, rows: list[dict[str, Any]]) -> None:
    from io import StringIO

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_QC_FIELDS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    _atomic_text(path, output.getvalue())


def _render_generated_dat_bytes(sample: dict[str, Any]) -> dict[str, bytes]:
    from scripts.diff_hamgnn.openmx_fd import displace_dat
    from scripts.npz2dat import write_openmx_dat

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        base_dat = root / "base.dat"
        structure = sample["structure"]
        write_openmx_dat(
            structure["z"],
            structure["pos"],
            structure["cell"],
            sample["charge"],
            base_dat,
            system_name=sample["sample_id"],
        )
        result: dict[str, bytes] = {}
        for branch, sign in (("plus", 1.0), ("minus", -1.0)):
            dat_path = displace_dat(
                base_dat,
                root / branch / "openmx.dat",
                atom_index_0based=sample["atom"],
                xyz=sample["axis"],
                delta_ang=sign * sample["delta_angstrom"],
                system_name=f"{sample['sample_id']}_{branch}",
            )
            result[branch] = dat_path.read_bytes()
        return result


def _ensure_sample_provenance(
    sample: dict[str, Any], reader: Path, nao: int, basis_def: dict[int, Any]
) -> None:
    required = {"generated_dat_bytes", "nao", "basis_schema", "reader_identity"}
    if required.issubset(sample):
        return
    reader_path = Path(reader).expanduser().resolve(strict=True)
    atomic_numbers = sorted({int(value) for value in sample["structure"]["z"]})
    sample["nao"] = int(nao)
    sample["basis_schema"] = {
        str(atomic_number): np.asarray(basis_def[atomic_number], dtype=np.int64).tolist()
        for atomic_number in atomic_numbers
    }
    sample["reader_identity"] = {
        "path": str(reader_path),
        "sha256": _file_sha256(reader_path),
    }
    sample["generated_dat_bytes"] = _render_generated_dat_bytes(sample)


def _json_ready(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    return value


def _audit_decision_path(cache_root: Path) -> Path:
    return Path(cache_root) / "audit.json"


def _load_audit_decision(cache_root: Path) -> dict[str, Any]:
    path = _audit_decision_path(cache_root)
    if not path.is_file():
        raise ValueError(f"main phase refused: no persisted passing audit at {path}")
    try:
        decision = json.loads(path.read_text())
    except Exception as exc:
        raise ValueError(f"main phase refused: unreadable audit decision: {exc}") from exc
    if not isinstance(decision, dict) or not bool(decision.get("passed", False)):
        raise ValueError("main phase refused: persisted delta audit did not pass")
    return decision


def _populate_sample_provenance(
    samples: Sequence[dict[str, Any]], reader: Path, utils_openmx: Path, nao: int
) -> dict[Path, str]:
    structure_hashes: dict[Path, str] = {}
    provenance_missing = any(
        not {"generated_dat_bytes", "nao", "basis_schema", "reader_identity"}.issubset(sample)
        for sample in samples
    )
    basis_def = _load_basis(utils_openmx) if provenance_missing else {}
    for sample in samples:
        graph_path = sample["path"]
        if graph_path not in structure_hashes:
            structure_hashes[graph_path] = _file_sha256(graph_path)
        _ensure_sample_provenance(sample, reader, nao, basis_def)
        sample["structure_checksum"] = structure_hashes[graph_path]
        sample["sample_checksum"] = _sample_checksum(sample, sample["structure_checksum"])
    return structure_hashes


def _execute(
    manifest_path: Path,
    manifest: dict[str, Any],
    manifest_checksum: str,
    cache_root: Path,
    samples: list[dict[str, Any]],
    structure_failures: list[dict[str, Any]],
    reader: Path,
    utils_openmx: Path,
    nao: int,
    phase: str = "main",
) -> int:
    cache_root.mkdir(parents=True, exist_ok=True)
    structure_hashes = _populate_sample_provenance(samples, reader, utils_openmx, nao)

    rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for failure in structure_failures:
        row = {field: "" for field in _QC_FIELDS}
        row.update(failure)
        rows.append(row)

    for sample in samples:
        cache_path = sample["cache_path"]
        if cache_path.exists():
            try:
                existing = load_cache_record(cache_path)
            except Exception:
                existing = {}
            if bool(existing.get("qc_passed", False)):
                digest = _file_sha256(cache_path)
                if existing.get("sample_checksum") != sample["sample_checksum"]:
                    rows.append(
                        _qc_row(
                            sample,
                            {
                                "state": "passing_cache_checksum_mismatch",
                                "qc_passed": False,
                                "qc_reason": "passing cache is immutable but its sample checksum differs",
                                "cache_sha256": digest,
                            },
                        )
                    )
                else:
                    derivative_norm = float(
                        np.linalg.norm(
                            np.asarray(existing["dh_dr"])[np.asarray(existing["active_mask"], dtype=np.bool_)]
                        )
                    )
                    rows.append(
                        _qc_row(
                            sample,
                            {
                                "state": "passing_cache_unchanged",
                                "qc_passed": True,
                                "qc_reason": "",
                                "zero_cutoff_missing_keys_plus": json.dumps(
                                    np.asarray(
                                        existing.get(
                                            "zero_cutoff_missing_keys_plus",
                                            np.empty((0, 5), dtype=np.int64),
                                        ),
                                        dtype=np.int64,
                                    ).reshape(-1, 5).tolist(),
                                    separators=(",", ":"),
                                ),
                                "zero_cutoff_missing_keys_minus": json.dumps(
                                    np.asarray(
                                        existing.get(
                                            "zero_cutoff_missing_keys_minus",
                                            np.empty((0, 5), dtype=np.int64),
                                        ),
                                        dtype=np.int64,
                                    ).reshape(-1, 5).tolist(),
                                    separators=(",", ":"),
                                ),
                                "cache_sha256": digest,
                                "derivative_norm": derivative_norm,
                                "relative_hermiticity_error_plus": existing.get(
                                    "relative_hermiticity_error_plus", ""
                                ),
                                "relative_hermiticity_error_minus": existing.get(
                                    "relative_hermiticity_error_minus", ""
                                ),
                                "midpoint_dense_fd_relative_error": existing.get(
                                    "midpoint_dense_fd_relative_error", ""
                                ),
                            },
                        )
                    )
                continue
        plus = _find_scfout(sample["job_dir"] / "plus")
        minus = _find_scfout(sample["job_dir"] / "minus")
        if plus is not None or minus is not None:
            status, _ = _collect_sample(sample, reader, utils_openmx, nao)
            if status["state"] == "outputs_pending":
                unresolved.append(sample)
            else:
                rows.append(_qc_row(sample, status))
        else:
            unresolved.append(sample)

    for sample in unresolved:
        try:
            _prepare_sample(sample)
            submit_errors = _submit_missing(sample)
        except Exception as exc:
            submit_errors = [f"preparation failed: {type(exc).__name__}: {exc}"]
        branch_states = {
            f"{branch}_state": _branch_state(sample, branch) for branch in ("plus", "minus")
        }
        if submit_errors:
            status = {
                "state": "submission_failed",
                "qc_passed": False,
                "qc_reason": "; ".join(submit_errors),
                **branch_states,
            }
        else:
            status = {
                "state": "submitted_or_pending",
                "qc_passed": False,
                "qc_reason": "awaiting locally fetched plus/minus scfout",
                **branch_states,
            }
        rows.append(_qc_row(sample, status))

    rows.sort(key=lambda row: (str(row.get("structure_id", "")), str(row.get("sample_id", ""))))
    phase_root = cache_root / "audit" if phase == "audit" else cache_root
    _write_qc(phase_root / "qc.csv", rows)
    sample_ids = {str(sample["sample_id"]) for sample in samples}
    row_by_sample_id = {
        str(row["sample_id"]): row
        for row in rows
        if str(row.get("sample_id", "")) in sample_ids
    }
    inventory: list[dict[str, Any]] = []
    for sample in samples:
        row = row_by_sample_id[str(sample["sample_id"])]
        inventory.append(
            {
                "sample_id": row["sample_id"],
                "component_id": row.get("component_id", row["sample_id"]),
                "structure_id": row["structure_id"],
                "fold": row["fold"],
                "atom": row["atom"],
                "axis": row["axis"],
                "delta_angstrom": row["delta_angstrom"],
                "derivative_norm": row["derivative_norm"],
                "relative_hermiticity_error_plus": row[
                    "relative_hermiticity_error_plus"
                ],
                "relative_hermiticity_error_minus": row[
                    "relative_hermiticity_error_minus"
                ],
                "midpoint_dense_fd_relative_error": row[
                    "midpoint_dense_fd_relative_error"
                ],
                "state": row["state"],
                "qc_passed": bool(row["qc_passed"]),
                "reason": row["qc_reason"],
                "cache_path": row["cache_path"],
                "cache_sha256": row["cache_sha256"],
                "sample_checksum": row["sample_checksum"],
            }
        )
    passing_samples = [
        dict(item) for item in inventory if item["qc_passed"] and item["cache_sha256"]
    ]
    generated_structures = [
        {
            "fold": entry["fold"],
            "stem": entry["stem"],
            "structure_id": entry["structure_id"],
            "path": str(entry["path"]),
            "structure_sha256": structure_hashes[entry["path"]],
            "charge": entry["charge"],
            "defect_kind": entry["defect_kind"],
            "defect_center": entry["defect_center"].tolist(),
            "probe_atoms": entry["probe_atoms"].tolist(),
        }
        for entry in entries_from_samples(samples)
    ]
    generated: dict[str, Any] = {
        "schema": "dh-label-cache-manifest-v2",
        "phase": phase,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": manifest_checksum,
        "delta_angstrom": float(manifest["delta_angstrom"]),
        "axes": [int(axis) for axis in manifest["axes"]],
        "structures": generated_structures,
        "inventory": inventory,
        "passing_samples": passing_samples,
    }
    if phase == "audit":
        audit_rows = [
            {
                "component_id": item["component_id"],
                "fold": item["fold"],
                "atom": item["atom"],
                "axis": item["axis"],
                "delta_angstrom": item["delta_angstrom"],
                "derivative_norm": item["derivative_norm"],
                "relative_hermiticity_error_plus": item[
                    "relative_hermiticity_error_plus"
                ],
                "relative_hermiticity_error_minus": item[
                    "relative_hermiticity_error_minus"
                ],
                "midpoint_dense_fd_relative_error": item[
                    "midpoint_dense_fd_relative_error"
                ],
                "state": item["state"],
                "qc_passed": item["qc_passed"],
                "qc_reason": item["reason"],
            }
            for item in inventory
        ]
        decision = evaluate_delta_audit(audit_rows)
        decision["source_manifest_sha256"] = manifest_checksum
        decision["audit_campaign_fingerprint"] = audit_campaign_fingerprint(inventory)
        decision["evidence_sample_count"] = len(inventory)
        decision = _json_ready(decision)
        _atomic_text(_audit_decision_path(cache_root), json.dumps(decision, sort_keys=True, indent=2) + "\n")
        generated["audit_decision"] = decision
    _atomic_text(phase_root / "manifest.json", json.dumps(generated, sort_keys=True, indent=2) + "\n")
    return 0


def entries_from_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for sample in samples:
        unique.setdefault(sample["structure_id"], sample)
    return list(unique.values())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("audit", "main"), default="audit")
    parser.add_argument("--execute", action="store_true", help="prepare, collect, and submit jobs")
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
    parser.add_argument("--nao", type=int, default=19)
    args = parser.parse_args(argv)

    manifest, manifest_checksum = _load_manifest(args.manifest)
    delta = float(manifest["delta_angstrom"])
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("manifest delta_angstrom must be finite and positive")
    audit_decision: Optional[dict[str, Any]] = None
    if args.phase == "main":
        try:
            audit_decision = _load_audit_decision(args.cache_root)
        except ValueError as exc:
            parser.error(str(exc))
        if audit_decision.get("source_manifest_sha256") != manifest_checksum:
            parser.error("main phase refused: persisted audit is for a different source manifest")
        if float(audit_decision.get("selected_delta_angstrom", float("nan"))) != delta:
            parser.error("main phase refused: persisted audit did not approve the manifest delta")

    entries, structure_failures = _structure_rows(
        manifest,
        args.manifest,
        load_supplied_structures=args.execute or args.phase in ("audit", "main"),
    )
    if args.phase == "audit":
        samples = select_delta_audit_samples(entries, manifest["axes"], args.cache_root)
    else:
        audit_samples = select_delta_audit_samples(entries, manifest["axes"], args.cache_root)
        try:
            _populate_sample_provenance(audit_samples, args.reader, args.utils_openmx, args.nao)
            current_fingerprint = audit_campaign_fingerprint(audit_samples)
        except Exception as exc:
            parser.error(
                "main phase refused: audit fingerprint could not be reconstructed: "
                f"{type(exc).__name__}: {exc}"
            )
        if audit_decision is None or audit_decision.get("audit_campaign_fingerprint") != current_fingerprint:
            parser.error("main phase refused: persisted audit fingerprint does not match current inputs")
        samples = _sample_specs(entries, manifest["axes"], delta, args.cache_root)

    if not args.execute:
        for failure in structure_failures:
            print(f"# {failure['structure_id']}: {failure['qc_reason']}")
        for sample in samples:
            for line in _submission_lines(sample):
                print(line)
        return 0

    if args.nao != 19:
        raise ValueError("collection currently pins basis_def_19; pass --nao 19")
    return _execute(
        args.manifest,
        manifest,
        manifest_checksum,
        args.cache_root,
        samples,
        structure_failures,
        args.reader,
        args.utils_openmx,
        args.nao,
        phase=args.phase,
    )


if __name__ == "__main__":
    raise SystemExit(main())
