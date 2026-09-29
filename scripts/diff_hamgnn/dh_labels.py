"""Derivative-label schema and topology-alignment helpers for the
charged-defect Hamiltonian response campaign.

Public API
----------
CampaignManaint : frozen dataclass wrapping the full manifest metadata.
DerivativeLabel : frozen dataclass for one derivative label.
derivative_from_pair : central-difference gradient estimate.
edge_keys : build (src, dst, sx, sy, sz) 5-tuples from topology arrays.
align_openmx_blocks_to_graph : reorder OpenMX blocks to match graph-edge order.
grouped_folds : read the YAML manifest and return fold -> list of stems.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CampaignManifest:
    """Frozen container for the full campaign manifest metadata.

    Carries global parameters (delta, probe count, axes, seed),
    fold-to-stem assignments, and the structure index mapping stem names
    to (charge, path) entries.
    """

    delta_angstrom: float
    probe_atoms_per_structure: int
    axes: Tuple[int, ...]
    seed: int
    folds: Dict[int, List[str]]
    structures: Dict[str, List[Dict[str, Any]]]


@dataclass(frozen=True)
class DerivativeLabel:
    """One derivative label for a specific atom, axis, and structure.

    Cache payload arrays are stored as float32; scalar metadata may be
    float64.  The ``h_mid`` and ``dh_dr`` fields hold the mid-point
    Hamiltonian (H averaged from the two displaced calls) and its
    central-difference gradient.
    """

    sample_id: str
    graph_path: Path
    stem: str
    charge: int
    fold: int
    atom: int
    axis: int
    delta_angstrom: float
    h_mid: np.ndarray
    dh_dr: np.ndarray
    active_mask: np.ndarray
    edge_index: np.ndarray
    cell_shift: np.ndarray


# ---------------------------------------------------------------------------
# Topology-safe helpers
# ---------------------------------------------------------------------------


def derivative_from_pair(
    h_plus: np.ndarray,
    h_minus: np.ndarray,
    delta_angstrom: float,
) -> np.ndarray:
    """Central-difference estimate of the Hamiltonian derivative dH/dr.

    Parameters
    ----------
    h_plus : np.ndarray
        Hamiltonian evaluated at r + delta.
    h_minus : np.ndarray
        Hamiltonian evaluated at r - delta.
    delta_angstrom : float
        Finite-difference step in Angstrom (must be positive).

    Returns
    -------
    np.ndarray
        (h_plus - h_minus) / (2 * delta_angstrom).  Dtype matches input.
    """
    if delta_angstrom <= 0.0:
        raise ValueError(
            f"delta_angstrom must be positive, got {delta_angstrom}"
        )
    return (np.asarray(h_plus) - np.asarray(h_minus)) / (2.0 * delta_angstrom)


def edge_keys(
    edge_index: np.ndarray,
    cell_shift: np.ndarray,
) -> List[Tuple[int, int, int, int, int]]:
    """Build ``(src, dst, sx, sy, sz)`` 5-tuples from topology arrays.

    Parameters
    ----------
    edge_index : np.ndarray
        Shape ``(2, n_edges)`` with source/destination atom indices.
    cell_shift : np.ndarray
        Integer lattice-vector shifts.  Either canonical ``(n_edges, 3)``
        (one shift row per edge, as returned by ``pack_hs_blocks()``) or the
        transposed ``(3, n_edges)`` layout (one shift column per edge) is
        accepted; the two are indistinguishable only for the square case.

    Returns
    -------
    list[tuple[int, int, int, int, int]]
        One 5-tuple per edge.
    """
    edge_array = np.asarray(edge_index)
    shift_array = np.asarray(cell_shift)
    if edge_array.ndim != 2 or edge_array.shape[0] != 2:
        raise ValueError(
            "edge_index must have shape (2, n_edges), got "
            f"{edge_array.shape}"
        )
    n_edges = edge_array.shape[1]
    if shift_array.ndim != 2:
        raise ValueError(f"cell_shift must be rank two, got {shift_array.shape}")
    if shift_array.shape == (3, n_edges):
        # Transposed layout: columns are the per-edge shifts.
        shifts = shift_array.T
    elif shift_array.shape == (n_edges, 3):
        shifts = shift_array
    else:
        raise ValueError(
            "cell_shift must have shape (n_edges, 3) or (3, n_edges) "
            f"matching edge_index; given {shift_array.shape} for {n_edges} edges"
        )
    return [
        (int(src), int(dst), int(sx), int(sy), int(sz))
        for (src, dst), (sx, sy, sz) in zip(edge_array.T, shifts)
    ]


def align_openmx_blocks_to_graph(
    blocks: np.ndarray,
    openmx_keys: List[Tuple[int, int, int, int, int]],
    graph_keys: List[Tuple[int, int, int, int, int]],
) -> np.ndarray:
    """Reorder OpenMX off-site blocks to match the graph-edge key order.

    Parameters
    ----------
    blocks : np.ndarray
        Off-site blocks in OpenMX native order.  Shape ``(n_edges, nao, nao)``
        or ``(n_edges, flat_size)``.
    openmx_keys : list of 5-tuples
        Keys for the *blocks* array, one per block.
    graph_keys : list of 5-tuples
        Target key order the caller expects (typically from HamGNN's
        ``edge_index`` / ``cell_shift``).

    Returns
    -------
    np.ndarray
        Blocks reordered so that slot ``i`` corresponds to ``graph_keys[i]``.

    Raises
    ------
    ValueError
        If the key sets differ in size or content, or if *openmx_keys*
        contains duplicate entries.
    """
    positions = {key: idx for idx, key in enumerate(openmx_keys)}
    if len(positions) != len(openmx_keys):
        raise ValueError(
            f"openmx_keys contains duplicate entries "
            f"({len(openmx_keys)} entries, {len(positions)} unique)"
        )
    if set(positions) != set(graph_keys):
        missing = set(graph_keys) - set(positions)
        extra = set(positions) - set(graph_keys)
        parts = []
        if missing:
            parts.append(f"missing from openmx: {sorted(missing)}")
        if extra:
            parts.append(f"extra in openmx: {sorted(extra)}")
        raise ValueError(
            "OpenMX and graph topologies differ: " + "; ".join(parts)
        )
    return np.asarray([blocks[positions[key]] for key in graph_keys])


def _validate_topology_keys(
    keys: List[Tuple[int, int, int, int, int]],
    name: str,
) -> None:
    """Require canonical, hashable five-integral-component edge keys."""
    for index, key in enumerate(keys):
        if not isinstance(key, tuple) or len(key) != 5:
            raise ValueError(
                f"{name}[{index}] must be a 5-tuple of integral components"
            )
        if any(
            isinstance(component, (bool, np.bool_))
            or not isinstance(component, Integral)
            for component in key
        ):
            raise ValueError(
                f"{name}[{index}] must contain exactly five integral components"
            )


def drop_openmx_extra_edges(
    blocks: np.ndarray,
    openmx_keys: List[Tuple[int, int, int, int, int]],
    graph_keys: List[Tuple[int, int, int, int, int]],
    threshold_ha: float = 1.0e-6,
) -> Optional[np.ndarray]:
    """Drop OpenMX-only low-norm edge blocks, returning blocks or None.

    The graph topology is authoritative for the label contract: OpenMX runs at
    the cutoff boundary occasionally contain a reciprocal pair of extra edges
    that the graph generator excluded (a critical-distance neighbor).  Their
    H blocks are weak-coupling noise (measured 2.2e-7 Ha for the GaAs
    campaign).  This helper drops every OpenMX-only occurrence whose block
    Frobenius norm is at most ``threshold_ha``, reordering the survivors to
    the graph key order.  It returns ``None`` if any OpenMX-only block is
    material (norm > threshold) — callers must fail loudly then, never
    silently drop real physics.

    Duplicate-key FIFO matching mirrors :func:`align_openmx_blocks_to_graph`.
    """
    block_array = np.asarray(blocks)
    positions = {key: idx for idx, key in enumerate(openmx_keys)}
    if len(positions) != len(openmx_keys):
        raise ValueError("openmx_keys contains duplicate entries")
    source_positions = {key: idx for idx, key in enumerate(graph_keys)}
    if len(source_positions) != len(graph_keys):
        raise ValueError("graph_keys contains duplicate entries")
    extra_keys = sorted(set(positions) - set(source_positions))
    if not extra_keys:
        return align_openmx_blocks_to_graph(block_array, openmx_keys, graph_keys)
    if len(block_array.shape) not in (2, 3):
        raise ValueError(
            f"blocks must have rank 2 or 3 with an edge dimension first, got "
            f"shape {block_array.shape}"
        )
    for key in extra_keys:
        block = block_array[positions[key]]
        if float(np.linalg.norm(block)) > threshold_ha:
            return None
    survivors = [key for key in openmx_keys if key not in set(extra_keys)]
    survivor_array = np.asarray([block_array[positions[key]] for key in survivors])
    return align_openmx_blocks_to_graph(survivor_array, survivors, graph_keys)


def align_openmx_blocks_to_graph_with_zero_cutoff(
    blocks: np.ndarray,
    openmx_keys: List[Tuple[int, int, int, int, int]],
    graph_keys: List[Tuple[int, int, int, int, int]],
    graph_reference_blocks: np.ndarray,
) -> np.ndarray:
    """Align a source-subset topology, zero-filling only guarded omissions.

    Unlike :func:`align_openmx_blocks_to_graph`, this explicit opt-in path
    permits graph-only block occurrences when every omitted occurrence has a
    reciprocal omitted partner and its graph-reference Frobenius norm is at
    most ``1e-6`` Ha. Duplicate keys are matched FIFO and retain their full
    multiplicity; OpenMX-only occurrences are never discarded.
    """
    block_array = np.asarray(blocks)
    reference_array = np.asarray(graph_reference_blocks)
    if block_array.ndim not in (2, 3):
        raise ValueError(
            "blocks must have rank 2 or 3 with an edge dimension first, got "
            f"shape {block_array.shape}"
        )
    if reference_array.ndim not in (2, 3):
        raise ValueError(
            "graph_reference_blocks must have rank 2 or 3 with an edge "
            f"dimension first, got shape {reference_array.shape}"
        )
    if block_array.shape[0] != len(openmx_keys):
        raise ValueError(
            "blocks and openmx_keys must have equal lengths, got "
            f"{block_array.shape[0]} and {len(openmx_keys)}"
        )
    if reference_array.shape[0] != len(graph_keys):
        raise ValueError(
            "graph_reference_blocks and graph_keys must have equal lengths, got "
            f"{reference_array.shape[0]} and {len(graph_keys)}"
        )
    if block_array.shape[1:] != reference_array.shape[1:]:
        raise ValueError(
            "blocks and graph_reference_blocks must have identical trailing "
            f"shapes, got {block_array.shape[1:]} and {reference_array.shape[1:]}"
        )

    _validate_topology_keys(openmx_keys, "openmx_keys")
    _validate_topology_keys(graph_keys, "graph_keys")
    positions: Dict[Tuple[int, int, int, int, int], deque[int]] = defaultdict(deque)
    for index, key in enumerate(openmx_keys):
        positions[key].append(index)

    order: List[int | None] = []
    missing_indices: List[int] = []
    for graph_index, key in enumerate(graph_keys):
        candidates = positions.get(key)
        if candidates:
            order.append(candidates.popleft())
        else:
            order.append(None)
            missing_indices.append(graph_index)

    native_only = [
        key
        for key, remaining in positions.items()
        for _ in range(len(remaining))
    ]
    if native_only:
        raise ValueError(
            "OpenMX topology contains occurrences absent from graph topology: "
            f"{native_only}"
        )

    missing_positions: Dict[
        Tuple[int, int, int, int, int], deque[int]
    ] = defaultdict(deque)
    for graph_index in missing_indices:
        missing_positions[graph_keys[graph_index]].append(graph_index)
    for graph_index in missing_indices:
        key = graph_keys[graph_index]
        queue = missing_positions[key]
        if not queue or queue[0] != graph_index:
            continue
        queue.popleft()
        src, dst, sx, sy, sz = key
        reciprocal = (dst, src, -sx, -sy, -sz)
        reciprocal_queue = missing_positions.get(reciprocal)
        if not reciprocal_queue:
            raise ValueError(
                "graph-only topology occurrence has no omitted reciprocal: "
                f"{key}"
            )
        reciprocal_queue.popleft()

    for graph_index in missing_indices:
        norm = float(np.linalg.norm(reference_array[graph_index]))
        if not np.isfinite(norm) or norm > 1.0e-6:
            raise ValueError(
                "graph-only reference block Frobenius norm exceeds the "
                f"inclusive 1e-6 Ha cutoff for {graph_keys[graph_index]}: {norm}"
            )

    aligned = np.zeros(
        (len(graph_keys), *block_array.shape[1:]), dtype=block_array.dtype
    )
    for graph_index, native_index in enumerate(order):
        if native_index is not None:
            aligned[graph_index] = block_array[native_index]
    return aligned


def grouped_folds(manifest_path: Path) -> Dict[int, List[str]]:
    """Read the campaign manifest and return fold -> list of stem names.

    Parameters
    ----------
    manifest_path : Path
        Path to ``config/dh_response_manifest.yaml``.

    Returns
    -------
    dict[int, list[str]]
        Mapping from fold number to its assigned stem names.
    """
    with open(manifest_path) as f:
        data: dict[str, Any] = yaml.safe_load(f)

    folds: Dict[int, List[str]] = {}
    for raw_key, fold_data in data["folds"].items():
        fold = int(raw_key)
        stems: List[str] = list(fold_data["stems"])
        folds[fold] = stems
    return folds
