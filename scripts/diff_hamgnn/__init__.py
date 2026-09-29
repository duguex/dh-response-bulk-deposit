"""Autodiff / EPC wrapper stack for **frozen** M4 HamGNN (not a model fork).

Loads pretrained ``hamgnn_pre_charge`` + ``HamGNN_out`` from HamGNN_v2.0;
weights stay frozen. Geometry derivatives use existing ``pos→edge→H`` ops plus
post-processing (HF, S providers, Zhong AO).

Phases:
  P0 smoke_p0_regression.py
  P1 smoke_k_hf.py
  P2 smoke_s_provider.py   (S-B full-AD edge RBF + S-A FD)
  P3 smoke_full_hf.py      (include_S full HF)
  P4 smoke_ao_correction.py
  P5 p5_alignment.py → docs/P5_ALIGNMENT_REPORT.md

Docs: docs/M4_BAND_AUTODIFF_PATH.md, docs/DFPT_S_R_ROADMAP.md
"""
from .ao_correction import ao_operator_derivative, matrix_element
from .assemble import (
    as_kpoints,
    assemble_gamma,
    assemble_k,
    hermitize,
    keep_orbital_indices,
    split_hon_hoff,
)
from .defaults import resolve_paths
from .differentiable import DifferentiableHamGNN, Eigenpairs, bootstrap_hamgnn
from .invariants import check_translation_invariance
from .s_provider import (
    DifferentiableEdgeSProvider,
    FiniteDifferenceSProvider,
    FrozenGraphSProvider,
    SBlocks,
    SProvider,
    ad_grad_S_sum,
)

__all__ = [
    "DifferentiableHamGNN",
    "Eigenpairs",
    "assemble_gamma",
    "assemble_k",
    "as_kpoints",
    "hermitize",
    "keep_orbital_indices",
    "split_hon_hoff",
    "bootstrap_hamgnn",
    "resolve_paths",
    "check_translation_invariance",
    "SProvider",
    "SBlocks",
    "FrozenGraphSProvider",
    "DifferentiableEdgeSProvider",
    "FiniteDifferenceSProvider",
    "ad_grad_S_sum",
    "ao_operator_derivative",
    "matrix_element",
]
