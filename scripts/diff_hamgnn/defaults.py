"""Pinned defaults for M4 autodiff regression (P0).

Stack pin — do not mix versions:
  tree:   HamGNN_v2.0 / HamGNN_v_1_0
  env:    hamgnn_v2.0_env
  net:    hamgnn_pre_charge + HamGNN_out
  ckpt:   charged_defect_hamiltoian.ckpt
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


def _home() -> Path:
    return Path.home()


def _repo_root() -> Path:
    # scripts/diff_hamgnn/defaults.py → repo root
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class M4Paths:
    hamgnn_root: Path
    config: Path
    graph: Path
    ckpt: Path

    def missing(self) -> List[Path]:
        out = []
        for p in (self.hamgnn_root, self.config, self.graph, self.ckpt):
            if not Path(p).exists():
                out.append(Path(p))
        return out


# Primary local paths (this workstation)
DEFAULT_HAMGNN_ROOT = _home() / "defect-ml-repro" / "HamGNN_v2.0"
DEFAULT_CONFIG = _repo_root() / "config" / "config_zenodo_test.yaml"
DEFAULT_GRAPH = _repo_root() / "data" / "npz" / "As_Ga_chg0.npz"
DEFAULT_CKPT = (
    _home()
    / "developing"
    / "HamGNN"
    / "examples"
    / "charged"
    / "charged_defect_hamiltoian.ckpt"
)

# Fallbacks (gpu_cluster / zenodo layout) — used only if primary missing
_FALLBACKS = {
    "graph": [
        _home() / "hamgnn-zenodo" / "models" / "hamgnn-q" / "As_Ga_chg0.npz",
        _home() / "slurm-4090-guide" / "data" / "npz" / "As_Ga_chg0.npz",
    ],
    "ckpt": [
        _home()
        / "hamgnn-zenodo"
        / "models"
        / "hamgnn-q"
        / "charged_defect_hamiltoian.ckpt",
        DEFAULT_CKPT,
    ],
    "config": [
        DEFAULT_CONFIG,
        _home() / "slurm-4090-guide" / "config" / "config_zenodo_test.yaml",
    ],
    "hamgnn_root": [
        DEFAULT_HAMGNN_ROOT,
        _home() / "defect-ml-repro" / "HamGNN_v2.0",
    ],
}

# Regression thresholds (P0 exit criteria)
THRESH_H_MEDIAN_REL = 0.15
THRESH_EPS_CORR = 0.95
THRESH_TRANSLATION_FORCE_SUM = 1e-3  # Ha/Å — sum_I F_I
THRESH_TRANSLATION_E_REL = 1e-5  # |E(R+Δ)-E(R)| / (|E|+eps)


def _first_existing(candidates: List[Path]) -> Path:
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return Path(candidates[0])


def resolve_paths(
    hamgnn_root: Optional[Path] = None,
    config: Optional[Path] = None,
    graph: Optional[Path] = None,
    ckpt: Optional[Path] = None,
) -> M4Paths:
    """Resolve paths with env overrides and fallbacks.

    Env (optional):
      DIFF_HAMGNN_ROOT, DIFF_HAMGNN_CONFIG, DIFF_HAMGNN_GRAPH, DIFF_HAMGNN_CKPT
    """
    root = Path(
        hamgnn_root
        or os.environ.get("DIFF_HAMGNN_ROOT", "")
        or _first_existing(_FALLBACKS["hamgnn_root"])
    )
    cfg = Path(
        config
        or os.environ.get("DIFF_HAMGNN_CONFIG", "")
        or _first_existing(_FALLBACKS["config"])
    )
    gr = Path(
        graph
        or os.environ.get("DIFF_HAMGNN_GRAPH", "")
        or _first_existing(_FALLBACKS["graph"])
    )
    ck = Path(
        ckpt
        or os.environ.get("DIFF_HAMGNN_CKPT", "")
        or _first_existing(_FALLBACKS["ckpt"])
    )
    return M4Paths(hamgnn_root=root, config=cfg, graph=gr, ckpt=ck)


def default_device() -> str:
    return os.environ.get("DIFF_HAMGNN_DEVICE", "cuda:0")


def argparse_add_m4_paths(ap) -> None:
    """Add standard --hamgnn-root/--config/--graph/--ckpt/--device to a parser."""
    paths = resolve_paths()
    ap.add_argument("--hamgnn-root", type=Path, default=paths.hamgnn_root)
    ap.add_argument("--config", type=Path, default=paths.config)
    ap.add_argument("--graph", type=Path, default=paths.graph)
    ap.add_argument("--ckpt", type=Path, default=paths.ckpt)
    ap.add_argument("--device", default=default_device())


def paths_as_dict(paths: M4Paths) -> Dict[str, str]:
    return {
        "hamgnn_root": str(paths.hamgnn_root),
        "config": str(paths.config),
        "graph": str(paths.graph),
        "ckpt": str(paths.ckpt),
    }
