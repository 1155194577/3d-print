"""Slice a finished STL into PLA G-code for the CUHK Library MakerSpace Creality K1C.

PrusaSlicer's CLI runs on the print server with server/profiles/creality_k1c_pla.ini. A model that
is too big, a missing slicer or a failed slice skips the G-code; the STL is still delivered.
"""

import os
import re
import shutil
import subprocess
import tempfile
import threading

PROFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles", "creality_k1c_pla.ini")
PRINTER = "Creality K1C (CUHK MakerSpace)"
FILAMENT = "PLA 1.75 mm"
BED_MM = (220.0, 220.0, 250.0)
TIMEOUT_S = float(os.environ.get("PRINT_SLICE_TIMEOUT_S", "300"))
_one_at_a_time = threading.Lock()  # each slice holds ~250 MB; the box caps the service at 750 MB

_MOVE = re.compile(rb"^G[01] [^;\n]*", re.M)
_AXIS = re.compile(rb"([XYZ])(-?\d+\.?\d*)")


def slicer():
    path = os.environ.get("PRUSA_SLICER") or shutil.which("prusa-slicer")
    return path if path and os.access(path, os.X_OK) else None


def _seconds(text):
    units = dict(d=86400, h=3600, m=60, s=1)
    return sum(int(n) * units[u] for n, u in re.findall(r"(\d+)([dhms])", text))


def _summary(gcode):
    tail = gcode[-20000:].decode("ascii", "replace")

    def value(name):
        match = re.search(rf"^; {re.escape(name)} = (.+)$", tail, re.M)
        return match.group(1).strip() if match else None

    time_text = value("estimated printing time (normal mode)")
    grams = value("total filament used [g]")
    metres = value("filament used [mm]")
    return dict(print_time=time_text, print_time_s=_seconds(time_text) if time_text else None,
                filament_g=float(grams) if grams else None,
                filament_m=round(float(metres) / 1000, 2) if metres else None)


def bounds(gcode):
    """[[min x, y, z], [max x, y, z]] over the moves that set an axis, in mm."""
    lo, hi = [float("inf")] * 3, [float("-inf")] * 3
    for move in _MOVE.findall(gcode):
        for axis, number in _AXIS.findall(move):
            i, v = b"XYZ".index(axis), float(number)
            lo[i], hi[i] = min(lo[i], v), max(hi[i], v)
    return [lo, hi]


def _fits(box):
    lo, hi = box
    return all(lo[i] >= 0 and hi[i] <= BED_MM[i] for i in range(3))


def slice_stl(stl, extents_mm, name):
    """(G-code bytes or None, report). `extents_mm` is the model's size; `name` its G-code filename."""
    base = dict(printer=PRINTER, filament=FILAMENT, layer_mm=0.2, filename=name)
    if any(e > b for e, b in zip(sorted(extents_mm[:2]) + [extents_mm[2]], sorted(BED_MM[:2]) + [BED_MM[2]])):
        return None, dict(base, skipped="bigger than the K1C's 220 × 220 × 250 mm build volume")
    binary = slicer()
    if binary is None:
        return None, dict(base, skipped="no slicer on this server")
    with tempfile.TemporaryDirectory(prefix="slice-") as tmp, _one_at_a_time:
        src, out = os.path.join(tmp, "model.stl"), os.path.join(tmp, "model.gcode")
        with open(src, "wb") as f:
            f.write(stl)
        cmd = [binary, "--export-gcode", "--load", PROFILE, "--center", f"{BED_MM[0] / 2:g},{BED_MM[1] / 2:g}",
               "--threads", "2", "-o", out, src]
        try:
            run = subprocess.run(cmd, capture_output=True, timeout=TIMEOUT_S, cwd=tmp,
                                 env=dict(os.environ, MALLOC_ARENA_MAX="2"))
        except subprocess.TimeoutExpired:
            return None, dict(base, skipped=f"slicing took over {TIMEOUT_S:g} s")
        if run.returncode != 0 or not os.path.exists(out):
            lines = (run.stderr or run.stdout).decode("utf-8", "replace").strip().splitlines()
            return None, dict(base, skipped="the slicer failed: " + (lines[-1] if lines else f"exit {run.returncode}"))
        with open(out, "rb") as f:
            gcode = f.read()
    box = bounds(gcode)
    if not _fits(box):
        return None, dict(base, skipped="the sliced toolpath leaves the 220 × 220 × 250 mm build volume")
    return gcode, dict(base, **_summary(gcode), bounds_mm=[[round(v, 1) for v in c] for c in box],
                       size_bytes=len(gcode))
