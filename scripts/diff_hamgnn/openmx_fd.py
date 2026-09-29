"""OpenMX helpers for external FD of S and band energies (Track A / P5).

- ``displace_dat`` / ``run_openmx_postprocess`` / ``scfout_S_frobenius``: P5 S-spot
- ``run_openmx_scf`` / ``parse_eigenvalues_out`` / ``band_energy_from_out``: #22 Track A
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def displace_dat(
    src_dat: Path,
    dst_dat: Path,
    atom_index_0based: int,
    xyz: int,
    delta_ang: float,
    data_path: Optional[str] = None,
    system_name: Optional[str] = None,
) -> Path:
    """Copy OpenMX .dat and shift one atom Cartesian coordinate (Angstrom)."""
    lines = Path(src_dat).read_text().splitlines(keepends=True)
    out: List[str] = []
    in_coords = False
    coord_i = 0
    for line in lines:
        stripped = line.strip()
        if data_path is not None and stripped.startswith("DATA.PATH"):
            out.append(f"DATA.PATH                      {data_path}\n")
            continue
        if system_name is not None and stripped.startswith("System.Name"):
            out.append(f"System.Name                   {system_name}\n")
            continue
        if stripped.startswith("<Atoms.SpeciesAndCoordinates"):
            in_coords = True
            out.append(line)
            continue
        if stripped.startswith("Atoms.SpeciesAndCoordinates>"):
            in_coords = False
            out.append(line)
            continue
        if in_coords and stripped and not stripped.startswith("#"):
            parts = stripped.split()
            if len(parts) >= 5 and parts[0].isdigit():
                if coord_i == atom_index_0based:
                    vals = list(parts)
                    vals[2 + xyz] = f"{float(parts[2 + xyz]) + delta_ang:.10f}"
                    out.append("  " + "  ".join(vals) + "\n")
                    coord_i += 1
                    continue
                coord_i += 1
        out.append(line)
    Path(dst_dat).parent.mkdir(parents=True, exist_ok=True)
    Path(dst_dat).write_text("".join(out))
    return Path(dst_dat)


def run_openmx_postprocess(
    work_dir: Path,
    dat_name: str = "openmx.dat",
    postprocess_bin: Optional[Path] = None,
    timeout_s: float = 900.0,
) -> Path:
    """Run openmx_postprocess; return path to overlap.scfout."""
    work_dir = Path(work_dir)
    bin_path = (
        postprocess_bin
        or Path.home() / "developing/openmx3.9/openmx_postprocess/openmx_postprocess"
    )
    if not bin_path.exists():
        raise FileNotFoundError(f"openmx_postprocess not found: {bin_path}")
    log = work_dir / "pp.out"
    with open(log, "w") as f:
        r = subprocess.run(
            [str(bin_path), dat_name],
            cwd=str(work_dir),
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
    if r.returncode != 0:
        raise RuntimeError(f"openmx_postprocess failed rc={r.returncode}; see {log}")
    scf = work_dir / "overlap.scfout"
    if not scf.exists():
        cands = list(work_dir.glob("*.scfout"))
        if not cands:
            raise FileNotFoundError(f"no scfout in {work_dir}")
        scf = cands[0]
    return scf


def run_openmx_scf(
    work_dir: Path,
    dat_name: str = "openmx.dat",
    openmx_bin: Optional[Path] = None,
    nprocs: int = 1,
    timeout_s: float = 3600.0,
) -> Path:
    """Run full OpenMX SCF; return path to ``*.out`` with eigenvalues."""
    work_dir = Path(work_dir)
    bin_path = openmx_bin or Path.home() / "repos/HamGNN/openmx_source_code/openmx"
    if not bin_path.exists():
        for cand in (
            Path.home() / "developing/openmx3.9/source/openmx",
            Path.home() / "developing/openmx3.9/work/openmx",
        ):
            if cand.exists():
                bin_path = cand
                break
    if not bin_path.exists():
        raise FileNotFoundError(f"openmx binary not found: {bin_path}")

    log = work_dir / "scf_stdout.log"
    cmd = [str(bin_path), dat_name]
    if nprocs > 1:
        cmd = ["mpirun", "-np", str(nprocs), str(bin_path), dat_name]
    with open(log, "w") as f:
        r = subprocess.run(
            cmd,
            cwd=str(work_dir),
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
    if r.returncode != 0:
        raise RuntimeError(f"openmx SCF failed rc={r.returncode}; see {log}")

    outs = sorted(work_dir.glob("*.out"))
    outs = [p for p in outs if p.stat().st_size > 1000]
    if not outs:
        if log.exists() and "Eigenvalues" in log.read_text(errors="ignore"):
            return log
        raise FileNotFoundError(f"no OpenMX .out with eigenvalues in {work_dir}")
    outs.sort(key=lambda p: p.stat().st_size, reverse=True)
    return outs[0]


def parse_eigenvalues_out(out_path: Path) -> dict:
    """Parse SCF KS eigenvalues (Hartree) from OpenMX ``.out``."""
    text = Path(out_path).read_text(errors="ignore")
    marker = "Eigenvalues (Hartree) for SCF KS-eq"
    idx = text.rfind(marker)
    if idx < 0:
        marker = "Eigenvalues (Hartree)"
        idx = text.rfind(marker)
    if idx < 0:
        raise ValueError(f"no Eigenvalues block in {out_path}")
    block = text[idx : idx + 500_000]

    chem_p = None
    m = re.search(r"Chemical Potential \(Hartree\)\s*=\s*([-\d.eE+]+)", block)
    if m:
        chem_p = float(m.group(1))
    homo = None
    m = re.search(r"HOMO\s*=\s*(\d+)", block)
    if m:
        homo = int(m.group(1))
    n_states = None
    m = re.search(r"Number of States\s*=\s*([-\d.eE+]+)", block)
    if m:
        n_states = float(m.group(1))

    evals_up: List[float] = []
    evals_down: List[float] = []
    expect = 1
    for line in block.splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[0].isdigit():
            if evals_up and expect > 10:
                break
            continue
        idx_i = int(parts[0])
        if idx_i != expect:
            if evals_up:
                break
            continue
        try:
            up = float(parts[1])
        except ValueError:
            if evals_up:
                break
            continue
        evals_up.append(up)
        if len(parts) >= 3:
            try:
                evals_down.append(float(parts[2]))
            except ValueError:
                pass
        expect += 1

    if not evals_up:
        raise ValueError(f"failed to parse eigenvalue rows in {out_path}")

    return {
        "evals_up": np.asarray(evals_up, dtype=np.float64),
        "evals_down": np.asarray(evals_down, dtype=np.float64) if evals_down else None,
        "chem_p": chem_p,
        "homo": homo,
        "n_states": n_states,
        "path": str(out_path),
    }


def band_energy_from_out(
    out_path: Path,
    bands: Union[slice, Sequence[int], None] = None,
    spin: str = "up",
) -> float:
    """Sum of KS eigenvalues in a band window (Hartree).

    Default ``bands=slice(10, 20)`` matches ``DifferentiableHamGNN.band_energy_hf``.
    """
    parsed = parse_eigenvalues_out(out_path)
    if spin == "down" and parsed["evals_down"] is not None:
        ev = parsed["evals_down"]
    else:
        ev = parsed["evals_up"]
    if bands is None:
        bands = slice(10, min(20, len(ev)))
    if isinstance(bands, slice):
        window = ev[bands]
    else:
        idx = np.asarray(list(bands), dtype=int)
        window = ev[idx]
    return float(np.sum(window))


def scfout_S_frobenius(scfout: Path) -> float:
    """Scalar ||S||_F from all OLP blocks (external FD target)."""
    import sys

    utils = Path.home() / "repos/HamGNN/utils_openmx"
    if str(utils) not in sys.path:
        sys.path.insert(0, str(utils))
    from read_openmx_dm import ScfOutReader

    r = ScfOutReader(str(scfout))
    r.read()
    ssq = 0.0

    def _acc(x):
        nonlocal ssq
        if x is None:
            return
        if isinstance(x, np.ndarray):
            ssq += float(np.sum(np.abs(x) ** 2))
        elif isinstance(x, list):
            for y in x:
                _acc(y)

    _acc(r.OLP)
    return float(np.sqrt(ssq))


def openmx_central_fd_band_energy(
    src_dat: Path,
    work_root: Path,
    atom: int,
    xyz: int,
    delta_ang: float = 0.001,
    bands: Union[slice, Sequence[int], None] = None,
    data_path: Optional[str] = None,
    openmx_bin: Optional[Path] = None,
    timeout_s: float = 3600.0,
    nprocs: int = 1,
) -> Tuple[float, dict]:
    """Central FD of OpenMX band-window energy w.r.t. one Cartesian component.

    Returns ``(dE/dR, meta)`` in Hartree/Å. Reuses ``openmx.out`` if already present.
    """
    work_root = Path(work_root)
    meta: dict = {"atom": atom, "xyz": xyz, "delta": delta_ang, "plus": {}, "minus": {}}
    energies = {}
    for tag, sign in (("plus", +1.0), ("minus", -1.0)):
        w = work_root / f"a{atom}_c{xyz}_{tag}"
        w.mkdir(parents=True, exist_ok=True)
        dat = w / "openmx.dat"
        displace_dat(
            src_dat,
            dat,
            atom_index_0based=atom,
            xyz=xyz,
            delta_ang=sign * delta_ang,
            data_path=data_path,
            system_name="openmx",
        )
        cached = w / "openmx.out"
        if (
            cached.exists()
            and cached.stat().st_size > 1000
            and "Eigenvalues" in cached.read_text(errors="ignore")
        ):
            out_path = cached
        else:
            out_path = run_openmx_scf(
                w,
                dat_name="openmx.dat",
                openmx_bin=openmx_bin,
                nprocs=nprocs,
                timeout_s=timeout_s,
            )
        e = band_energy_from_out(out_path, bands=bands)
        energies[tag] = e
        meta[tag] = {"out": str(out_path), "E": e}
    fd = (energies["plus"] - energies["minus"]) / (2.0 * delta_ang)
    meta["fd"] = float(fd)
    return float(fd), meta
