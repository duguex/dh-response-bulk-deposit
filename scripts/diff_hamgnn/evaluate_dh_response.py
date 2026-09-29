#!/usr/bin/env python3
"""Evaluate frozen M4 pointwise Hamiltonians and selected dH/dR components.

The evaluator is deliberately read-only with respect to the label cache.  It
accepts only a complete, passing main-phase cache whose source-manifest digest
and selected finite-difference delta are bound to a persisted passing audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260715
G0_RELATIVE_ERROR_THRESHOLD = 0.50
G0_REQUIRED_FOLDS = 4
G0_TOTAL_FOLDS = 5
G0_REQUIRED_DEFECT_CLASSES = 2
POINTWISE_H_NORMAL_MAX_MEV = 4.68
HARTREE_TO_MEV = 27_211.386245988

_ROW_FIELDS = (
    "sample_id",
    "structure_id",
    "fold",
    "stem",
    "charge",
    "defect_class",
    "atom",
    "axis",
    "delta_angstrom",
    "active_element_count",
    "pointwise_h_mae_mev",
    "dh_mae_mev_per_angstrom",
    "relative_frobenius_error",
    "frobenius_cosine_similarity",
    "derivative_norm_ratio",
)
_METRIC_FIELDS = (
    "pointwise_h_mae_mev",
    "dh_mae_mev_per_angstrom",
    "relative_frobenius_error",
    "frobenius_cosine_similarity",
    "derivative_norm_ratio",
)
_REQUIRED_CACHE_FIELDS = frozenset(
    {
        "h_mid",
        "dh_dr",
        "active_mask",
        "edge_index",
        "cell_shift",
        "delta_angstrom",
        "qc_passed",
        "qc_reason",
        "sample_checksum",
        "structure_checksum",
        "sample_id",
        "structure_id",
        "atom",
        "axis",
    }
)


def _metric_vectors(
    predicted: np.ndarray,
    reference: np.ndarray,
    active_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(predicted)
    target = np.asarray(reference)
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {prediction.shape} does not match reference shape {target.shape}"
        )
    if active_mask is None:
        mask = np.ones(target.shape, dtype=np.bool_)
    else:
        mask = np.asarray(active_mask, dtype=np.bool_)
        if mask.shape != target.shape:
            raise ValueError(
                f"active_mask shape {mask.shape} does not match matrix shape {target.shape}"
            )
    if not np.any(mask):
        raise ValueError("active_mask selects no elements")
    prediction_active = np.asarray(prediction[mask])
    target_active = np.asarray(target[mask])
    if not np.isfinite(prediction_active).all() or not np.isfinite(target_active).all():
        raise ValueError("metrics require finite active elements")
    return prediction_active, target_active


def relative_frobenius_error(
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    active_mask: np.ndarray | None = None,
) -> float:
    """Return ``||predicted-reference||F / (||reference||F + 1e-12)``."""
    prediction, target = _metric_vectors(predicted, reference, active_mask)
    return float(np.linalg.norm(prediction - target) / (np.linalg.norm(target) + 1.0e-12))


def frobenius_cosine_similarity(
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    active_mask: np.ndarray | None = None,
) -> float:
    """Return the active-element Frobenius cosine, zero-safe for either norm."""
    prediction, target = _metric_vectors(predicted, reference, active_mask)
    prediction_norm = float(np.linalg.norm(prediction))
    target_norm = float(np.linalg.norm(target))
    if prediction_norm == 0.0 or target_norm == 0.0:
        return 0.0
    similarity = np.vdot(target, prediction) / (target_norm * prediction_norm)
    value = float(np.real(similarity))
    if not np.isfinite(value):
        raise ValueError("Frobenius cosine is not finite")
    return value


def grouped_bootstrap_summary(
    values: Sequence[float] | np.ndarray,
    stems: Sequence[str] | np.ndarray,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Summarize values and a median bootstrap that resamples whole stems."""
    sample_values = np.asarray(values, dtype=np.float64)
    sample_stems = np.asarray(stems)
    if sample_values.ndim != 1 or sample_stems.ndim != 1:
        raise ValueError("values and stems must be one-dimensional")
    if len(sample_values) == 0 or len(sample_values) != len(sample_stems):
        raise ValueError("values and stems must have the same non-zero length")
    if not np.isfinite(sample_values).all():
        raise ValueError("bootstrap values must be finite")
    if int(n_resamples) != n_resamples or int(n_resamples) <= 0:
        raise ValueError("n_resamples must be a positive integer")
    ordered_stems = np.unique(sample_stems)
    if len(ordered_stems) == 0:
        raise ValueError("bootstrap requires at least one stem")
    rows_by_stem = {stem: sample_values[sample_stems == stem] for stem in ordered_stems}
    rng = np.random.default_rng(int(seed))
    boot_medians = np.empty(int(n_resamples), dtype=np.float64)
    for draw_index in range(int(n_resamples)):
        sampled_stems = rng.choice(ordered_stems, size=len(ordered_stems), replace=True)
        sampled_rows = np.concatenate([rows_by_stem[stem] for stem in sampled_stems])
        boot_medians[draw_index] = float(np.median(sampled_rows))
    q25, q75 = np.percentile(sample_values, [25.0, 75.0])
    ci_low, ci_high = np.percentile(boot_medians, [2.5, 97.5])
    return {
        "median": float(np.median(sample_values)),
        "mean": float(np.mean(sample_values)),
        "iqr": float(q75 - q25),
        "ci95": [float(ci_low), float(ci_high)],
        "n_values": int(len(sample_values)),
        "n_stems": int(len(ordered_stems)),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def g0_verdict(
    *,
    grouped_median_relative_error: float,
    fold_median_relative_errors: Sequence[float],
    defect_class_median_relative_errors: Mapping[str, float],
    pointwise_h_mae_normal: bool,
) -> dict[str, Any]:
    """Apply the immutable preregistered G0 phenomenon gate."""
    global_error = float(grouped_median_relative_error)
    fold_errors = [float(value) for value in fold_median_relative_errors]
    class_errors = {str(key): float(value) for key, value in defect_class_median_relative_errors.items()}
    if not np.isfinite(global_error):
        raise ValueError("grouped median relative error must be finite")
    if len(fold_errors) != G0_TOTAL_FOLDS or not np.isfinite(fold_errors).all():
        raise ValueError("G0 requires exactly five finite fold median relative errors")
    if not class_errors or not np.isfinite(list(class_errors.values())).all():
        raise ValueError("G0 requires finite defect-class median relative errors")
    qualifying_folds = sum(value > G0_RELATIVE_ERROR_THRESHOLD for value in fold_errors)
    qualifying_classes = sorted(
        key for key, value in class_errors.items() if value > G0_RELATIVE_ERROR_THRESHOLD
    )
    global_passed = global_error > G0_RELATIVE_ERROR_THRESHOLD
    folds_passed = qualifying_folds >= G0_REQUIRED_FOLDS
    classes_passed = len(qualifying_classes) >= G0_REQUIRED_DEFECT_CLASSES
    pointwise_passed = bool(pointwise_h_mae_normal)
    passed = bool(global_passed and folds_passed and classes_passed and pointwise_passed)
    return {
        "gate": "G0",
        "passed": passed,
        "verdict": "PASS" if passed else "NO-GO",
        "threshold": G0_RELATIVE_ERROR_THRESHOLD,
        "threshold_operator": ">",
        "grouped_median_relative_error": global_error,
        "global_relative_error_passed": global_passed,
        "fold_median_relative_errors": fold_errors,
        "qualifying_folds": int(qualifying_folds),
        "required_qualifying_folds": G0_REQUIRED_FOLDS,
        "total_folds": G0_TOTAL_FOLDS,
        "fold_condition_passed": folds_passed,
        "defect_class_median_relative_errors": class_errors,
        "qualifying_defect_classes": qualifying_classes,
        "required_qualifying_defect_classes": G0_REQUIRED_DEFECT_CLASSES,
        "defect_class_condition_passed": classes_passed,
        "pointwise_h_mae_normal": pointwise_passed,
        "pointwise_h_condition_passed": pointwise_passed,
    }


def model_component_fd(model: Any, atom: int, axis: int, delta: float) -> Any:
    """Compute one model dH/dR component by central difference, without a Jacobian."""
    atom_index = int(atom)
    axis_index = int(axis)
    step = float(delta)
    if axis_index not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    if atom_index < 0 or atom_index >= int(model.pos0.shape[0]):
        raise ValueError(f"atom index {atom_index} is out of range")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("delta must be finite and positive")
    plus = model.pos0.clone()
    minus = model.pos0.clone()
    plus[atom_index, axis_index] += step
    minus[atom_index, axis_index] -= step
    return (model.forward_H(plus) - model.forward_H(minus)) / (2.0 * step)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing {description}: {path}")
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise ValueError(f"unreadable {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and set(text) <= set("0123456789abcdef")


def _unique_rows(rows: Any, name: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"cache manifest {name} must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("sample_id"):
            raise ValueError(f"cache manifest {name} contains an invalid row")
        sample_id = str(row["sample_id"])
        if sample_id in indexed:
            raise ValueError(f"cache manifest {name} duplicates sample {sample_id!r}")
        indexed[sample_id] = row
    return indexed


def _manifest_stem_folds(source: dict[str, Any]) -> dict[str, int]:
    folds = source.get("folds")
    if not isinstance(folds, dict):
        raise ValueError("source manifest folds must be a mapping")
    result: dict[str, int] = {}
    for fold_key, specification in folds.items():
        fold = int(fold_key)
        if not isinstance(specification, dict) or not isinstance(specification.get("stems"), list):
            raise ValueError(f"source manifest fold {fold} has no stems list")
        for stem_value in specification["stems"]:
            stem = str(stem_value)
            if stem in result:
                raise ValueError(f"source manifest duplicates stem {stem!r}")
            result[stem] = fold
    if sorted(set(result.values())) != list(range(G0_TOTAL_FOLDS)):
        raise ValueError("source manifest must define folds 0 through 4")
    return result


def _source_structure_contract(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    structures = source.get("structures")
    if not isinstance(structures, dict):
        raise ValueError("source manifest structures must be a mapping")
    stem_folds = _manifest_stem_folds(source)
    if set(structures) != set(stem_folds):
        raise ValueError("source manifest structures and grouped stems differ")
    result: dict[str, dict[str, Any]] = {}
    for stem, rows in structures.items():
        if not isinstance(rows, list):
            raise ValueError(f"source manifest structure {stem!r} must be a list")
        for row in rows:
            if not isinstance(row, dict) or "path" not in row or "charge" not in row:
                raise ValueError(f"source manifest structure {stem!r} has an invalid row")
            structure_id = Path(str(row["path"])).stem
            if structure_id in result:
                raise ValueError(f"source manifest duplicates structure id {structure_id!r}")
            result[structure_id] = {
                "stem": str(stem),
                "fold": int(stem_folds[str(stem)]),
                "charge": int(row["charge"]),
                "path": str(row["path"]),
            }
    return result


def _load_and_validate_campaign(
    manifest_path: Path, cache_root: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    try:
        import yaml
    except ImportError as exc:
        raise ValueError("PyYAML is required to read the campaign manifest") from exc

    manifest_path = manifest_path.expanduser().resolve(strict=True)
    cache_root = cache_root.expanduser().resolve(strict=True)
    source_bytes = manifest_path.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    source = yaml.safe_load(source_bytes)
    if not isinstance(source, dict):
        raise ValueError("source manifest must be a mapping")
    for field in ("delta_angstrom", "probe_atoms_per_structure", "axes", "folds", "structures"):
        if field not in source:
            raise ValueError(f"source manifest missing required field {field!r}")
    delta = float(source["delta_angstrom"])
    axes = [int(axis) for axis in source["axes"]]
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("source manifest delta_angstrom must be finite and positive")
    if axes != [0, 1, 2]:
        raise ValueError("source manifest axes must be exactly [0, 1, 2]")
    if int(source["probe_atoms_per_structure"]) != 4:
        raise ValueError("source manifest must select exactly four probe atoms per structure")

    audit = _read_json(cache_root / "audit.json", "delta-audit decision")
    required_audit = {
        "schema": "dh-delta-audit-v1",
        "passed": True,
        "qc_passed": True,
        "source_manifest_sha256": source_sha,
        "component_count": 20,
        "required_passing_components": 16,
        "evidence_sample_count": 60,
    }
    for key, expected in required_audit.items():
        if audit.get(key) != expected:
            raise ValueError(
                f"delta-audit decision field {key!r} is {audit.get(key)!r}, expected {expected!r}"
            )
    if float(audit.get("selected_delta_angstrom", float("nan"))) != delta:
        raise ValueError("delta-audit decision did not accept the source-manifest delta")
    if int(audit.get("passing_components", -1)) < int(audit["required_passing_components"]):
        raise ValueError("delta-audit decision has too few passing components")
    if not _valid_sha256(audit.get("audit_campaign_fingerprint")):
        raise ValueError("delta-audit decision lacks a valid campaign fingerprint")

    audit_manifest = _read_json(cache_root / "audit" / "manifest.json", "audit cache manifest")
    if audit_manifest.get("schema") != "dh-label-cache-manifest-v2" or audit_manifest.get("phase") != "audit":
        raise ValueError("audit cache manifest has the wrong schema or phase")
    if audit_manifest.get("source_manifest_sha256") != source_sha:
        raise ValueError("audit cache manifest is for a different source manifest")
    if audit_manifest.get("audit_decision") != audit:
        raise ValueError("persisted audit decision differs from audit-manifest evidence")

    generated = _read_json(cache_root / "manifest.json", "main cache manifest")
    if generated.get("schema") != "dh-label-cache-manifest-v2" or generated.get("phase") != "main":
        raise ValueError("main cache manifest has the wrong schema or phase")
    if generated.get("source_manifest_sha256") != source_sha:
        raise ValueError("main cache manifest is for a different source manifest")
    if float(generated.get("delta_angstrom", float("nan"))) != delta:
        raise ValueError("main cache manifest delta differs from the accepted audit delta")
    if [int(axis) for axis in generated.get("axes", [])] != axes:
        raise ValueError("main cache manifest axes differ from the source manifest")

    source_structures = _source_structure_contract(source)
    generated_rows = generated.get("structures")
    if not isinstance(generated_rows, list):
        raise ValueError("main cache manifest structures must be a list")
    structures: dict[str, dict[str, Any]] = {}
    expected_ids: set[str] = set()
    for row in generated_rows:
        if not isinstance(row, dict) or not row.get("structure_id"):
            raise ValueError("main cache manifest contains an invalid structure row")
        structure_id = str(row["structure_id"])
        if structure_id in structures:
            raise ValueError(f"main cache manifest duplicates structure {structure_id!r}")
        if structure_id not in source_structures:
            raise ValueError(f"cache structure {structure_id!r} is absent from source manifest")
        contract = source_structures[structure_id]
        for key in ("stem", "fold", "charge"):
            if row.get(key) != contract[key]:
                raise ValueError(f"cache structure {structure_id!r} has wrong {key}")
        probes = [int(atom) for atom in row.get("probe_atoms", [])]
        if len(probes) != 4 or len(set(probes)) != 4 or min(probes) < 0:
            raise ValueError(f"cache structure {structure_id!r} lacks four unique probe atoms")
        if not _valid_sha256(row.get("structure_sha256")):
            raise ValueError(f"cache structure {structure_id!r} lacks a valid SHA-256")
        graph_path = Path(str(row.get("path", ""))).expanduser()
        if not graph_path.is_absolute():
            repository_path = _REPO / graph_path
            graph_path = repository_path if repository_path.exists() else manifest_path.parent / graph_path
        graph_path = graph_path.resolve(strict=True)
        if _sha256(graph_path) != row["structure_sha256"]:
            raise ValueError(f"graph checksum mismatch for structure {structure_id!r}")
        normalized = dict(row)
        normalized["path"] = graph_path
        normalized["probe_atoms"] = probes
        structures[structure_id] = normalized
        for atom in probes:
            for axis in axes:
                expected_ids.add(f"{structure_id}_atom{atom:03d}_axis{axis}")
    if set(structures) != set(source_structures):
        missing = sorted(set(source_structures) - set(structures))
        raise ValueError(f"main cache manifest is missing structures: {missing}")
    expected_count = len(source_structures) * 4 * len(axes)
    if expected_count != 540 or len(expected_ids) != expected_count:
        raise ValueError(f"campaign must contain exactly 540 unique labels, found {len(expected_ids)}")

    inventory = _unique_rows(generated.get("inventory"), "inventory")
    passing = _unique_rows(generated.get("passing_samples"), "passing_samples")
    if set(inventory) != expected_ids:
        missing = sorted(expected_ids - set(inventory))
        extra = sorted(set(inventory) - expected_ids)
        raise ValueError(f"cache inventory is incomplete (missing={missing[:5]}, extra={extra[:5]})")
    if set(passing) != expected_ids:
        missing = sorted(expected_ids - set(passing))
        extra = sorted(set(passing) - expected_ids)
        raise ValueError(f"passing label set is incomplete (missing={missing[:5]}, extra={extra[:5]})")
    for sample_id in sorted(expected_ids):
        inventory_row = inventory[sample_id]
        passing_row = passing[sample_id]
        if not bool(inventory_row.get("qc_passed")) or not bool(passing_row.get("qc_passed")):
            raise ValueError(f"label {sample_id!r} did not pass cache QC")
        for key in ("component_id", "structure_id", "fold", "atom", "axis", "sample_checksum", "cache_sha256"):
            if inventory_row.get(key) != passing_row.get(key):
                raise ValueError(f"inventory and passing provenance differ for {sample_id!r}: {key}")
        if float(passing_row.get("delta_angstrom", float("nan"))) != delta:
            raise ValueError(f"label {sample_id!r} uses a non-accepted delta")
        if not _valid_sha256(passing_row.get("sample_checksum")) or not _valid_sha256(
            passing_row.get("cache_sha256")
        ):
            raise ValueError(f"label {sample_id!r} lacks valid checksums")
    return source, structures, passing


def _load_cache_sample(
    cache_root: Path,
    sample_id: str,
    passing_row: Mapping[str, Any],
    structure: Mapping[str, Any],
    delta: float,
) -> dict[str, Any]:
    from scripts.diff_hamgnn.build_dh_label_cache import load_cache_record

    cache_path = cache_root / "samples" / f"{sample_id}.npz"
    if not cache_path.is_file():
        raise ValueError(f"passing cache file is missing: {cache_path}")
    if _sha256(cache_path) != passing_row["cache_sha256"]:
        raise ValueError(f"cache file checksum mismatch for {sample_id!r}")
    record = load_cache_record(cache_path)
    missing = sorted(_REQUIRED_CACHE_FIELDS - set(record))
    if missing:
        raise ValueError(f"cache label {sample_id!r} is missing fields: {missing}")
    if not bool(record["qc_passed"]) or str(record["qc_reason"]):
        raise ValueError(f"cache label {sample_id!r} is not a clean passing label")
    expected_scalars = {
        "sample_id": sample_id,
        "structure_id": str(passing_row["structure_id"]),
        "atom": int(passing_row["atom"]),
        "axis": int(passing_row["axis"]),
        "sample_checksum": str(passing_row["sample_checksum"]),
        "structure_checksum": str(structure["structure_sha256"]),
    }
    for key, expected in expected_scalars.items():
        actual = record[key]
        if isinstance(expected, int):
            actual = int(actual)
        else:
            actual = str(actual)
        if actual != expected:
            raise ValueError(f"cache label {sample_id!r} has mismatched field {key!r}")
    if float(record["delta_angstrom"]) != delta:
        raise ValueError(f"cache label {sample_id!r} has a non-accepted delta")
    h_mid = np.asarray(record["h_mid"], dtype=np.float64)
    derivative = np.asarray(record["dh_dr"], dtype=np.float64)
    mask = np.asarray(record["active_mask"], dtype=np.bool_)
    if h_mid.ndim != 2 or derivative.shape != h_mid.shape or mask.shape != h_mid.shape:
        raise ValueError(f"cache label {sample_id!r} has incompatible matrix shapes")
    if not np.any(mask) or not np.isfinite(h_mid[mask]).all() or not np.isfinite(derivative[mask]).all():
        raise ValueError(f"cache label {sample_id!r} has invalid active elements")
    return record


def _tensor_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _validate_graph_alignment(model: Any, record: Mapping[str, Any], sample_id: str) -> np.ndarray:
    from scripts.diff_hamgnn.build_dh_label_cache import _active_block_mask

    model_edges = _tensor_numpy(model.edge_index).astype(np.int64, copy=False)
    model_shifts = _tensor_numpy(model.cell_shift).astype(np.int64, copy=False)
    cache_edges = np.asarray(record["edge_index"], dtype=np.int64)
    cache_shifts = np.asarray(record["cell_shift"], dtype=np.int64)
    if not np.array_equal(model_edges, cache_edges) or not np.array_equal(model_shifts, cache_shifts):
        raise ValueError(f"model graph order differs from cache label {sample_id!r}")
    atomic_numbers = _tensor_numpy(model.data.z).astype(np.int64, copy=False)
    expected_mask = _active_block_mask(atomic_numbers, model_edges, model.nao, model.basis_def)
    cache_mask = np.asarray(record["active_mask"], dtype=np.bool_)
    if not np.array_equal(expected_mask, cache_mask):
        raise ValueError(f"active mask differs from graph-ordered model basis for {sample_id!r}")
    return cache_mask


def _summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stems = np.asarray([str(row["stem"]) for row in rows])
    return {
        field: grouped_bootstrap_summary(
            np.asarray([float(row[field]) for row in rows], dtype=np.float64), stems
        )
        for field in _METRIC_FIELDS
    }


def _group_tables(rows: Sequence[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
    values = sorted({row[field] for row in rows}, key=lambda value: (str(type(value)), str(value)))
    table: list[dict[str, Any]] = []
    for value in values:
        selected = [row for row in rows if row[field] == value]
        table.append(
            {
                field: value,
                "n_components": len(selected),
                "n_structures": len({str(row["structure_id"]) for row in selected}),
                "n_stems": len({str(row["stem"]) for row in selected}),
                "metrics": _summaries(selected),
            }
        )
    return table


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_ROW_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in _ROW_FIELDS})
    return output.getvalue()


def _format_metric(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.6g}"


def _markdown_table(table: Sequence[Mapping[str, Any]], grouping: str) -> str:
    lines = [
        f"| {grouping} | Components | Stems | H MAE (meV) | dH MAE (meV/Å) | Relative error | Cosine | Norm ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        metrics = row["metrics"]
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row[grouping]),
                    str(row["n_components"]),
                    str(row["n_stems"]),
                    _format_metric(metrics["pointwise_h_mae_mev"]["median"]),
                    _format_metric(metrics["dh_mae_mev_per_angstrom"]["median"]),
                    _format_metric(metrics["relative_frobenius_error"]["median"]),
                    _format_metric(metrics["frobenius_cosine_similarity"]["median"]),
                    _format_metric(metrics["derivative_norm_ratio"]["median"]),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _report(metrics: Mapping[str, Any]) -> str:
    overall = metrics["overall"]
    verdict = metrics["g0_verdict"]
    overall_row = {
        "scope": "all",
        "n_components": metrics["sample_count"],
        "n_stems": metrics["stem_count"],
        "metrics": overall,
    }
    condition_rows = (
        ("Global median relative error > 0.50", verdict["global_relative_error_passed"]),
        ("At least 4/5 folds have median relative error > 0.50", verdict["fold_condition_passed"]),
        ("At least two defect classes have median relative error > 0.50", verdict["defect_class_condition_passed"]),
        (
            f"Pointwise H MAE <= {POINTWISE_H_NORMAL_MAX_MEV:.2f} meV",
            verdict["pointwise_h_condition_passed"],
        ),
    )
    lines = [
        "# Frozen M4 Hamiltonian-Response Benchmark",
        "",
        "## Exact G0 verdict",
        "",
        f"**G0 verdict: {verdict['verdict']}**",
        "",
        f"The preregistered relative-error threshold is immutable and strict: `> {verdict['threshold']:.2f}`.",
        "",
        "| Condition | Result |",
        "|---|---|",
    ]
    lines.extend(f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in condition_rows)
    lines.extend(
        [
            "",
            "## Overall",
            "",
            _markdown_table([overall_row], "scope"),
            "",
            "All intervals are percentile 95% intervals from 10,000 whole-stem bootstrap resamples with seed 20260715.",
            "",
            "## Fold breakdown",
            "",
            _markdown_table(metrics["by_fold"], "fold"),
            "",
            "## Charge breakdown",
            "",
            _markdown_table(metrics["by_charge"], "charge"),
            "",
            "## Defect-class breakdown",
            "",
            _markdown_table(metrics["by_defect_class"], "defect_class"),
            "",
        ]
    )
    return "\n".join(lines)


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    cache_root = args.cache_root.expanduser().resolve(strict=True)
    config = args.config.expanduser().resolve(strict=True)
    checkpoint = args.ckpt.expanduser().resolve(strict=True)
    hamgnn_root = args.hamgnn_root.expanduser().resolve(strict=True)
    source, structures, passing = _load_and_validate_campaign(manifest_path, cache_root)
    delta = float(source["delta_angstrom"])

    import torch
    from scripts.diff_hamgnn import DifferentiableHamGNN

    if str(args.device).startswith("cuda"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
    model = DifferentiableHamGNN.from_files(
        config, checkpoint, hamgnn_root, device=args.device, freeze_weights=True
    )
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("frozen-M4 evaluator refuses a model with trainable parameters")

    rows: list[dict[str, Any]] = []
    pointwise_absolute_sum = 0.0
    pointwise_active_count = 0
    with torch.no_grad():
        for structure_id in sorted(structures):
            structure = structures[structure_id]
            model.set_graph(structure["path"])
            model_mid = _tensor_numpy(model.forward_H(model.pos0)).astype(np.float64, copy=False)
            structure_h_counted = False
            sample_ids = sorted(
                sample_id
                for sample_id, row in passing.items()
                if str(row["structure_id"]) == structure_id
            )
            if len(sample_ids) != 12:
                raise ValueError(f"structure {structure_id!r} does not have exactly 12 passing labels")
            for sample_id in sample_ids:
                passing_row = passing[sample_id]
                record = _load_cache_sample(
                    cache_root, sample_id, passing_row, structure, delta
                )
                mask = _validate_graph_alignment(model, record, sample_id)
                target_mid = np.asarray(record["h_mid"], dtype=np.float64)
                target_derivative = np.asarray(record["dh_dr"], dtype=np.float64)
                if model_mid.shape != target_mid.shape:
                    raise ValueError(
                        f"model Hamiltonian shape {model_mid.shape} differs from label {sample_id!r} shape {target_mid.shape}"
                    )
                model_derivative = _tensor_numpy(
                    model_component_fd(
                        model, int(passing_row["atom"]), int(passing_row["axis"]), delta
                    )
                ).astype(np.float64, copy=False)
                if model_derivative.shape != target_derivative.shape:
                    raise ValueError(f"model derivative shape differs from label {sample_id!r}")
                prediction_active, target_active = _metric_vectors(
                    model_derivative, target_derivative, mask
                )
                model_mid_active, target_mid_active = _metric_vectors(model_mid, target_mid, mask)
                if not structure_h_counted:
                    pointwise_absolute_sum += float(np.sum(np.abs(model_mid_active - target_mid_active)))
                    pointwise_active_count += int(mask.sum())
                    structure_h_counted = True
                target_norm = float(np.linalg.norm(target_active))
                prediction_norm = float(np.linalg.norm(prediction_active))
                norm_ratio = prediction_norm / target_norm if target_norm > 0.0 else 0.0
                row = {
                    "sample_id": sample_id,
                    "structure_id": structure_id,
                    "fold": int(structure["fold"]),
                    "stem": str(structure["stem"]),
                    "charge": int(structure["charge"]),
                    "defect_class": str(structure["defect_kind"]),
                    "atom": int(passing_row["atom"]),
                    "axis": int(passing_row["axis"]),
                    "delta_angstrom": delta,
                    "active_element_count": int(mask.sum()),
                    "pointwise_h_mae_mev": float(
                        np.mean(np.abs(model_mid_active - target_mid_active)) * HARTREE_TO_MEV
                    ),
                    "dh_mae_mev_per_angstrom": float(
                        np.mean(np.abs(prediction_active - target_active)) * HARTREE_TO_MEV
                    ),
                    "relative_frobenius_error": relative_frobenius_error(
                        model_derivative, target_derivative, active_mask=mask
                    ),
                    "frobenius_cosine_similarity": frobenius_cosine_similarity(
                        model_derivative, target_derivative, active_mask=mask
                    ),
                    "derivative_norm_ratio": float(norm_ratio),
                }
                if not np.isfinite([float(row[field]) for field in _METRIC_FIELDS]).all():
                    raise ValueError(f"non-finite metric for label {sample_id!r}")
                rows.append(row)
    rows.sort(key=lambda row: str(row["sample_id"]))
    if len(rows) != 540 or pointwise_active_count <= 0:
        raise ValueError("evaluation did not produce the complete 540-label campaign")

    overall = _summaries(rows)
    by_fold = _group_tables(rows, "fold")
    by_charge = _group_tables(rows, "charge")
    by_class = _group_tables(rows, "defect_class")
    pointwise_h_mae_mev = pointwise_absolute_sum / pointwise_active_count * HARTREE_TO_MEV
    fold_errors = [
        float(row["metrics"]["relative_frobenius_error"]["median"]) for row in by_fold
    ]
    class_errors = {
        str(row["defect_class"]): float(row["metrics"]["relative_frobenius_error"]["median"])
        for row in by_class
    }
    verdict = g0_verdict(
        grouped_median_relative_error=float(overall["relative_frobenius_error"]["median"]),
        fold_median_relative_errors=fold_errors,
        defect_class_median_relative_errors=class_errors,
        pointwise_h_mae_normal=pointwise_h_mae_mev <= POINTWISE_H_NORMAL_MAX_MEV,
    )
    verdict["pointwise_h_mae_mev"] = float(pointwise_h_mae_mev)
    verdict["pointwise_h_normal_max_mev"] = POINTWISE_H_NORMAL_MAX_MEV
    metrics = {
        "schema": "frozen-m4-dh-response-v1",
        "sample_count": len(rows),
        "structure_count": len(structures),
        "stem_count": len({str(row["stem"]) for row in rows}),
        "delta_angstrom": delta,
        "bootstrap": {"n_resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "pointwise_h_mae_mev_active_elements": float(pointwise_h_mae_mev),
        "overall": overall,
        "by_fold": by_fold,
        "by_charge": by_charge,
        "by_defect_class": by_class,
        "g0_verdict": verdict,
        "provenance": {
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": _sha256(manifest_path),
            "cache_manifest": str(cache_root / "manifest.json"),
            "cache_manifest_sha256": _sha256(cache_root / "manifest.json"),
            "audit_decision": str(cache_root / "audit.json"),
            "audit_decision_sha256": _sha256(cache_root / "audit.json"),
            "config": str(config),
            "config_sha256": _sha256(config),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "hamgnn_root": str(hamgnn_root),
            "device": str(args.device),
            "model_load_meta": dict(model._load_meta),
        },
    }
    return {"rows": rows, "metrics": metrics}


def _parser() -> argparse.ArgumentParser:
    from scripts.diff_hamgnn.defaults import default_device, resolve_paths

    defaults = resolve_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=defaults.config)
    parser.add_argument("--ckpt", type=Path, default=defaults.ckpt)
    parser.add_argument("--hamgnn-root", type=Path, default=defaults.hamgnn_root)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = _evaluate(args)
        output_dir = args.output_dir.expanduser().resolve()
        rows_path = output_dir / "frozen_m4_rows.csv"
        metrics_path = output_dir / "frozen_m4_metrics.json"
        report_path = output_dir / "FROZEN_M4_REPORT.md"
        _atomic_text(rows_path, _csv_text(result["rows"]))
        _atomic_text(metrics_path, json.dumps(result["metrics"], sort_keys=True, indent=2) + "\n")
        _atomic_text(report_path, _report(result["metrics"]))
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"rows={rows_path}")
    print(f"metrics={metrics_path}")
    print(f"report={report_path}")
    print(f"G0={result['metrics']['g0_verdict']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
