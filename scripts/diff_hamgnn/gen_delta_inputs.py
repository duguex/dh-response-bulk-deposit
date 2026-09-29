#!/usr/bin/env python3
# Archive note (frozen 2026-09-25; as-run copy of
#   /mnt/shared/work/dh_response_bulk_labels_deltasub/gen_delta_inputs.py).
# Generates the 16 OpenMX input sets (plus/minus displacement at
#   delta in {0.0005, 0.002} A for four test-fold components) that
#   delta_floor_analysis.py reads; part of the label truncation audit behind
#   the Richardson floor quoted in the Track B manuscript.
"""Generate delta-Richardson subset OpenMX inputs from existing bulk label jobs.

For each chosen (structure, atom, axis) component, take the campaign base.dat
(midpoint geometry, Angstrom) and emit openmx.dat displaced by +-delta along
the axis for delta in {0.0005, 0.002}. Output layout mirrors the original
campaign so crisp can submit each run dir directly:

  <root>/jobs/<comp>/<name>/openmx.dat   with System.Name = <name>

No new midpoint (h_mid) runs needed: base midpoint matrices come from the
existing delta=0.001 campaign cache.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

CACHE = Path("/mnt/shared/work/dh_response_bulk_labels_v1/jobs")
ROOT = Path("/mnt/shared/work/dh_response_bulk_labels_deltasub/jobs")

# (component dir in campaign cache, probe atom 0-based, axis) — test-fold only
COMPONENTS = [
    ("Bulk04_chg0_atom060_axis1", 60, 1),
    ("Bulk04_chg0_atom028_axis0", 28, 0),
    ("Bulk19_chg0_atom007_axis2", 7, 2),
    ("Bulk19_chg0_atom032_axis1", 32, 1),
]
DELTAS = [0.0005, 0.002]

COORD_RE = re.compile(r"^\s*(\d+)\s+(\w+)\s+(\S+)\s+(\S+)\s+(\S+)")


def emit(base_text: str, atom0: int, axis: int, delta: float, sign: int, name: str, out_dir: Path) -> None:
    lines = base_text.splitlines(keepends=True)
    in_block = False
    target = atom0 + 1  # OpenMX atom numbering is 1-based
    hit = False
    for i, ln in enumerate(lines):
        if "<Atoms.SpeciesAndCoordinates" in ln:
            in_block = True
            continue
        if in_block and "Atoms.SpeciesAndCoordinates>" in ln:
            break
        if in_block:
            m = COORD_RE.match(ln)
            if m and int(m.group(1)) == target:
                x, y, z = float(m.group(3)), float(m.group(4)), float(m.group(5))
                shift = sign * delta
                if axis == 0:
                    x += shift
                elif axis == 1:
                    y += shift
                else:
                    z += shift
                rest = ln[m.end():]
                lines[i] = f"{int(m.group(1)):>5}  {m.group(2):<8} {x:>12.6f} {y:>12.6f} {z:>12.6f}" + rest
                hit = True
    if not hit:
        raise SystemExit(f"atom {target} not found in coordinate block")
    text = "".join(lines)
    text = re.sub(r"^System\.Name\s+\S+", f"System.Name                     {name}", text, count=1, flags=re.M)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "openmx.dat").write_text(text)
    print(f"wrote {out_dir/'openmx.dat'}  (atom {target} axis {axis} shift {sign*delta:+.4f} A)")


def main() -> None:
    for comp, atom0, axis in COMPONENTS:
        base = CACHE / comp / "base.dat"
        if not base.is_file():
            raise SystemExit(f"missing {base}")
        base_text = base.read_text()
        stem = comp  # e.g. Bulk04_chg0_atom060_axis1
        for delta in DELTAS:
            tag = f"d{str(delta).replace('0.','')}"  # d0005 / d002
            for sign, sname in ((+1, "p"), (-1, "m")):
                name = f"{stem}_{tag}{sname}"
                emit(base_text, atom0, axis, delta, sign, name, ROOT / comp / name)
    print("done")


if __name__ == "__main__":
    main()
