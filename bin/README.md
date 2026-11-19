# `triposr` — photo → printable 3D model

A one-command wrapper around TripoSR: feed it a photo, get a watertight STL sized for your printer.

```bash
triposr chair.jpg                      # → chair.stl, 200 mm max dimension
triposr chair.jpg --height 800         # exactly 800 mm tall
triposr mug.png -o mugs/mug.stl --size 90
triposr a.jpg b.jpg c.jpg              # one model per image
triposr shoe.jpg --mc-resolution 320   # finer geometry (slower)
triposr part.jpg --fit-bed 220x220x250 # shrink to fit a Creality K1C plate
```

Already on your `PATH` via the `~/.local/bin/triposr` symlink (and the
`/Users/johnny/Desktop/3d_printing/bin` entry in `~/.zshrc`). In a shell started
before the fix, run `source ~/.zshrc` or open a new terminal. You can always run
it directly as `/Users/johnny/Desktop/3d_printing/bin/triposr`.

## Python environment

The wrapper runs TripoSR from the conda env **`triposr`** (Python 3.11), so you
never need to activate anything — and conda's own `base` (Python 3.14) can't
interfere. Interpreter selection is, in order:

1. `$TRIPOSR_PYTHON`, if set
2. `~/miniconda3/envs/triposr/bin/python` ← normal path
3. `research/triposr_env311/bin/python` (older venv, fallback only)

Rebuild it from scratch with:

```bash
conda create -n triposr python=3.11 -y
conda run -n triposr pip install -r research/triposr_requirements.txt
conda run -n triposr pip install "git+https://github.com/tatsy/torchmcubes.git"
```

`torchmcubes` is a source build (marching cubes) and is not on PyPI — it must be
compiled in the target env. On macOS it builds CPU-only, which is what TripoSR
uses under MPS. Verified set: torch 2.15.0 (MPS), trimesh 5.2.1, pymeshfix
0.18.1, rembg 2.0.87, transformers 4.35.0, numpy 2.4.6.

## What it does

1. **TripoSR** turns the image into a 3D mesh on your Mac's GPU (MPS), ~30–90 s
2. **pymeshfix** closes marching-cubes holes and drops floating fragments
3. The bottom 3% is flattened so the model sits flat on the bed
4. A **base plate is fused** with a boolean union (helps multi-leg objects stand)
5. The result is scaled to real millimetres and exported as STL

## Options

| flag | meaning |
|---|---|
| `--height MM` | scale so the model is exactly this tall |
| `--size MM` | scale so the largest dimension is this (default 200) |
| `--fit-bed WxDxH` | shrink only if it doesn't fit a printer bed |
| `--no-base` | skip the base plate (furniture and flat-bottomed parts often need it) |
| `--base-thickness MM` | plate thickness, default 3 |
| `--mc-resolution N` | 256 default; 320–512 = more detail, more time/memory |
| `--chunk-size N` | lower it if you run out of memory |
| `--texture` | bake a UV texture (keep with `--format obj`/`glb` to see colors) |
| `--no-remove-bg` | input already has a clean gray background — skips rembg |
| `--foreground-ratio F` | how much of the frame the subject should fill (0.85) |
| `--format stl\|obj\|glb` | output format, default stl |
| `--keep-obj` | keep TripoSR's raw mesh next to the output |
| `--quiet` | print only the output paths (good for scripts) |

## Reading the report

```
  1. chair.stl
     307.8 × 543.1 × 800.5 mm   (93,648 faces, height → 800 mm, merged 5 parts, pymeshfix, base fused)
     watertight: ✓ yes   boundary edges: 0   non-manifold: 0   parts: 1
     volume: 17,527 cm³  (~21,734 g PLA solid)
     overhang faces: 23%  → supports recommended
```

- **watertight / parts: 1** — safe to slice as-is
- **overhang faces > 15%** — expect to enable supports in the slicer
- **volume** — multiply by ~1.24 for solid PLA grams; enable 15–20% infill to cut it ~4×

## Notes

- Model weights live in `research/hf_cache` and the background-removal model in
  `research/rembg_cache` — first run downloads ~1 GB, later runs are offline.
- Single-image output is a *plausible* object, not a measurement. For a faithful
  copy of one specific object, prefer many photos + photogrammetry.
- Batch mode reuses one process, so 5 images cost ~5× one image.
