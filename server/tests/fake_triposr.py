#!/usr/bin/env python3
"""Stand-in for bin/triposr, used by the API tests.

Emits the same milestone lines the real pipeline writes, at ~1/1000 the wall
clock, and produces a tiny closed tetrahedron so preview rendering still works.
This keeps the test suite at a couple of seconds instead of ~90 s per case.
"""

import sys
import time
from pathlib import Path

TETRA = """solid fake
 facet normal 0 0 -1
  outer loop
   vertex 0 0 0
   vertex 0 10 0
   vertex 12 0 0
  endloop
 endfacet
 facet normal 0 -1 0
  outer loop
   vertex 0 0 0
   vertex 12 0 0
   vertex 0 0 8
  endloop
 endfacet
 facet normal -1 0 0
  outer loop
   vertex 0 0 0
   vertex 0 0 8
   vertex 0 10 0
  endloop
 endfacet
 facet normal 1 1 1
  outer loop
   vertex 12 0 0
   vertex 0 10 0
   vertex 0 0 8
  endloop
 endfacet
endsolid fake
"""

LINES = [
    "[00:00:01] TripoSR: 1 image(s) → 3D  (device=mps, mc-res=256)",
    "2026-01-01 00:00:01,000 - INFO - Initializing model ...",
    "2026-01-01 00:00:02,000 - INFO - Initializing model finished in 1000.00ms.",
    "2026-01-01 00:00:02,010 - INFO - Processing images ...",
    "2026-01-01 00:00:03,000 - INFO - Processing images finished in 990.00ms.",
    "2026-01-01 00:00:03,010 - INFO - Running model ...",
    "2026-01-01 00:00:04,000 - INFO - Running model finished in 990.00ms.",
    "2026-01-01 00:00:04,010 - INFO - Extracting mesh ...",
    "2026-01-01 00:00:05,000 - INFO - Extracting mesh finished in 990.00ms.",
    "2026-01-01 00:00:05,010 - INFO - Exporting mesh ...",
    "2026-01-01 00:00:06,000 - INFO - Exporting mesh finished in 990.00ms.",
    "",
    "  1. fake.stl",
    "     12.0 × 10.0 × 8.0 mm   (4 faces, max dim → 12 mm, pymeshfix)",
    "     watertight: ✓ yes   boundary edges: 0   non-manifold: 0   parts: 1",
    "     volume: 160 cm³  (~198 g PLA solid)",
    "     overhang faces: 30%  → supports recommended",
    "",
    "  ✓ done — 1 model(s): %s",
]


def main() -> int:
    args = sys.argv[1:]
    out = None
    if "-o" in args:
        out = Path(args[args.index("-o") + 1])
    if out is None:
        print("fake: missing -o", file=sys.stderr)
        return 2
    # Hold the long stages briefly so a polling client is guaranteed to see
    # them, mirroring the real pipeline where each stage takes seconds.
    holds = {"Running model": 0.40, "Processing images": 0.20, "Extracting mesh": 0.20}
    for line in LINES:
        print(line % out if "%s" in line else line, flush=True)
        time.sleep(next((secs for key, secs in holds.items() if key in line), 0.01))
    out.write_text(TETRA)
    return 0


if __name__ == "__main__":
    sys.exit(main())
