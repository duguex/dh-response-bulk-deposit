"""AD-safe dense Hamiltonian / overlap assembly (Γ and general k).

Critical: do NOT use advanced indexing assignment ``H[j, i] = ...``.
That path breaks autograd (AD ≪ FD). Use ``index_add`` instead.

Bloch phase convention matches ``band_cal.py`` / OpenMX:
  phase_e = exp(2πi · cell_shift_e · k_frac)
which is equivalent to band_cal's
  exp(2πi · nbr_shift · (k_frac @ inv(cell).T))
because nbr_shift = cell_shift @ cell.
"""
from __future__ import annotations

import math
from typing import Tuple, Union

import torch


def assemble_gamma(
    Hon: torch.Tensor,
    Hoff: torch.Tensor,
    edge_index: torch.Tensor,
    nao: int,
) -> torch.Tensor:
    """Assemble dense AO matrix at Γ from on-site + off-site blocks.

    Parameters
    ----------
    Hon : (natoms, nao*nao)
    Hoff : (nedges, nao*nao)
    edge_index : (2, nedges) with rows (src=j, dst=i) matching HamGNN convention
    nao : orbitals per site (padded)

    Returns
    -------
    H : (natoms*nao, natoms*nao)
    """
    natoms = Hon.shape[0]
    j, i = edge_index[0], edge_index[1]
    Honb = Hon.reshape(natoms, nao, nao)
    Hoffb = Hoff.reshape(-1, nao, nao)
    blocks = Hon.new_zeros(natoms * natoms, nao, nao)
    on_idx = torch.arange(natoms, device=Hon.device) * (natoms + 1)
    blocks = blocks.index_add(0, on_idx, Honb)
    off_idx = j * natoms + i
    blocks = blocks.index_add(0, off_idx, Hoffb)
    H4 = blocks.view(natoms, natoms, nao, nao)
    return H4.permute(0, 2, 1, 3).reshape(natoms * nao, natoms * nao)


def bloch_phase(
    cell_shift: torch.Tensor,
    k_frac: torch.Tensor,
) -> torch.Tensor:
    """Per-edge Bloch phase exp(2πi k_frac·cell_shift).

    Parameters
    ----------
    cell_shift : (nedges, 3) integer lattice translations
    k_frac : (3,) fractional crystal coordinates

    Returns
    -------
    phase : (nedges,) complex128
    """
    k = k_frac.to(dtype=torch.float64, device=cell_shift.device).reshape(3)
    cs = cell_shift.to(dtype=torch.float64, device=k.device)
    # (nedges,)
    dot = (cs * k).sum(dim=-1)
    return torch.exp(2j * math.pi * dot)


def assemble_k(
    Hon: torch.Tensor,
    Hoff: torch.Tensor,
    edge_index: torch.Tensor,
    cell_shift: torch.Tensor,
    k_frac: torch.Tensor,
    nao: int,
) -> torch.Tensor:
    """Assemble dense H(k) or S(k) with AD-safe index_add + Bloch phase.

    Parameters
    ----------
    Hon, Hoff : real blocks (same layout as assemble_gamma)
    cell_shift : (nedges, 3)
    k_frac : (3,) fractional k

    Returns
    -------
    Hk : complex128 (natoms*nao, natoms*nao)
    """
    natoms = Hon.shape[0]
    j, i = edge_index[0], edge_index[1]
    phase = bloch_phase(cell_shift, k_frac)  # (nedges,) complex128

    Honb = Hon.reshape(natoms, nao, nao).to(dtype=torch.complex128)
    Hoffb = Hoff.reshape(-1, nao, nao).to(dtype=torch.complex128)
    Hoffb = Hoffb * phase[:, None, None]

    blocks = Honb.new_zeros(natoms * natoms, nao, nao)
    on_idx = torch.arange(natoms, device=Hon.device) * (natoms + 1)
    blocks = blocks.index_add(0, on_idx, Honb)
    off_idx = j * natoms + i
    blocks = blocks.index_add(0, off_idx, Hoffb)
    H4 = blocks.view(natoms, natoms, nao, nao)
    return H4.permute(0, 2, 1, 3).reshape(natoms * nao, natoms * nao)


def hermitize(M: torch.Tensor) -> torch.Tensor:
    """(M + M†)/2 for real or complex matrices."""
    if M.is_complex():
        return 0.5 * (M + M.conj().transpose(-2, -1))
    return 0.5 * (M + M.transpose(-2, -1))


def split_hon_hoff(
    H_flat: torch.Tensor, n_atoms: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split Model hamiltonian output into on-site / off-site blocks."""
    return H_flat[:n_atoms], H_flat[n_atoms:]


def keep_orbital_indices(
    z: torch.Tensor,
    nao: int,
    basis_def: dict,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Mask padded AO slots using HamGNN_out.basis_def (Z → active AO indices)."""
    device = device or z.device
    basis = torch.zeros(99, nao, device=device)
    for Z, idx in basis_def.items():
        basis[int(Z), list(idx)] = 1.0
    m = basis[z.long()]
    return torch.where(m.reshape(-1) > 0)[0]


def as_kpoints(
    kpoints: Union[torch.Tensor, list, tuple, None],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Normalize k-points to (nk, 3) float64 fractional coords. Default Γ only."""
    if kpoints is None:
        k = torch.zeros(1, 3, dtype=torch.float64)
    else:
        k = torch.as_tensor(kpoints, dtype=torch.float64)
        if k.ndim == 1:
            k = k.view(1, 3)
        if k.shape[-1] != 3:
            raise ValueError(f"kpoints last dim must be 3, got {tuple(k.shape)}")
    if device is not None:
        k = k.to(device)
    return k
