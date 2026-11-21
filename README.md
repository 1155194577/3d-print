# 3d-print — photo → printable STL

A small toolkit that turns a photograph into a watertight, correctly-scaled STL
for a 3D printer, plus a preview renderer for inspecting the result.

One command turns a single photograph into a watertight, correctly-scaled STL for a
3D printer. Wraps [TripoSR](https://github.com/VAST-AI-Research/TripoSR) (single-image
→ 3D on Apple MPS/CUDA/CPU) and adds the mesh work that makes the result printable:
repair, flatten, base plate, real-world scaling.

```bash
triposr photo.jpg                    # → photo.stl, 200 mm max dimension
triposr photo.jpg --height 120       # exactly 120 mm tall
triposr photo.jpg --fit-bed 220x220x250
triposr scan.usdz                    # LiDAR scan (USDZ/OBJ/PLY, metres): real size, no TripoSR
```

## Contents

| Path | What it is |
|---|---|
| `bin/triposr` | Shell wrapper: pins the interpreter, sets model caches, execs the pipeline |
| `tools/img2mesh.py` | The pipeline: TripoSR call → scale → flatten → repair → base plate → export |
| `tools/preview_stl.py` | Optional: render an STL to three orthographic PNG views for a quick look |
| `server/` | FastAPI web app: upload a photo in the browser, watch progress, download the STL |
| `server/print_api.py` | `POST /print`: LiDAR mesh or photo + height in, STL out (contract in `server/README.md`) |
| `deploy/modal_app.py` | Deploys the web app and `/print` on Modal (serverless GPU) |
| `research/triposr/` | Upstream TripoSR source (`run.py`, `tsr/`), MIT licensed |
| `research/triposr_requirements.txt` | Exact pinned dependency set that is known to work |
| `setup.sh` | Builds the conda env `triposr` from scratch |
| `examples/sample_can.jpg` | Sample input for a smoke test |

Not included: model weights (~2.9 GB, downloaded on first run) and the
background-remover model (~1 GB). Nothing binary is vendored.

## Requirements

- macOS on Apple silicon (MPS), or Linux with CUDA, or CPU (slow)
- conda (Miniconda/Anaconda)
- ~4 GB free disk for env + weights, and network on first run

## Install

```bash
./setup.sh                    # creates conda env `triposr` and installs everything
```

Then run it either by path:

```bash
./bin/triposr examples/sample_can.jpg
```

...or put it on `PATH` (mind the quotes — a stray space here is a known footgun):

```bash
echo 'export PATH="'"$PWD"'/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc
```

`setup.sh` also supports `./setup.sh --link`, which symlinks the wrapper into
`~/.local/bin`. The wrapper resolves symlinks correctly, so either entry point works.

Using a different conda root? Override at runtime:

```bash
TRIPOSR_PYTHON=/opt/conda/envs/triposr/bin/python triposr photo.jpg
# or
CONDA_PREFIX_DIR=/opt/conda/envs/triposr triposr photo.jpg
```

## Linux and GPU hosts (AutoDL, Colab-like images)

These images usually ship a conda whose channels are pointed at a mirror that
refuses the host's IP, so `conda create` dies with something like:

```
UnavailableInvalidChannel: HTTP 403 FORBIDDEN for channel anaconda/pkgs/main
<https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main>
```

That is a mirror problem, not a project problem. `setup.sh` handles two cases:

```bash
./setup.sh --venv        # skip conda entirely, build .venv from the image's python
./setup.sh --conda       # force conda (use with -c conda-forge when needed)
./setup.sh               # auto: conda, then fall back to .venv
```

On a China-hosted box, mirror everything:

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export GITHUB_MIRROR=https://ghproxy.com/          # for the torchmcubes clone
export HF_ENDPOINT=https://hf-mirror.com           # for model weights
./setup.sh --venv
```

CUDA notes: the PyPI `torch` wheel for linux-64 is a CUDA build, so `img2mesh.py`
auto-selects `cuda` and no `--device` flag is needed. `torchmcubes` compiles
against the CUDA toolkit when `nvcc` is present, otherwise it builds CPU-only
(slower marching cubes, same output). Passthrough env from the shell matters for
the web app too — it inherits `HF_HOME`, `TRIPOSR_PYTHON`, etc.

## Web app (FastAPI)

```
server/
├── app.py              job queue, SSE progress, upload validation, STL/preview routes
├── static/index.html   single page: drag-and-drop, live stages, preview, download
└── tests/              API tests driven by a fake pipeline (no GPU needed)
```

Run it:

```bash
pip install -r server/requirements-web.txt
python -m uvicorn server.app:app --host 0.0.0.0 --port 6006
```

| Route | Purpose |
|---|---|
| `GET /` | uploader page |
| `POST /api/jobs` | multipart image upload → `202 {id}` |
| `GET /api/jobs/{id}/events` | Server-Sent Events: progress, stage, metrics |
| `GET /api/jobs/{id}/model` | finished STL |
| `GET /api/jobs/{id}/preview.png` | three-view render |
| `GET /api/health` | liveness and whether the pipeline binary is present |
| `POST /print` | synchronous: LiDAR mesh, or photo + `height_mm`, → STL; see [`server/README.md`](server/README.md) |

Design notes: jobs run **one at a time** — TripoSR loads a ~1.5 GB model onto the
GPU per invocation, so parallel jobs would thrash memory rather than speed up.
Uploads are capped (`TRIPOSR_MAX_UPLOAD_MB`, default 20) and sniffed by magic
number, so a renamed binary is rejected before it reaches the pipeline. Jobs and
their files expire after `TRIPOSR_JOB_TTL_MIN` (default 120).

```bash
python -m pytest server/tests -q      # no GPU needed
```

## Docker

```bash
docker build -t 3d-print .
docker run --gpus all -p 8000:8000 -e PRINT_API_TOKEN=... -v 3d-print-models:/models 3d-print
```

Serves the web app and `POST /print` on port 8000. It runs on CPU when you leave out `--gpus all`. The weights (~2.7 GB) download into `/models` on first use; `--build-arg BAKE_WEIGHTS=1` bakes them into the image instead. On every PR and push, CI (`.github/workflows/docker.yml`) runs the tests, builds the image and smoke-tests it. Pushes to `main` and `v*` tags also publish it to `ghcr.io/1155194577/3d-print` (`latest`, `main`, `sha-…`, version).

## What it does, stage by stage

1. **TripoSR** — image → mesh (vertex colours, 256³ marching cubes) on the GPU, ~30–90 s
2. **Scale** — to `--height`, `--size`, or `--fit-bed`, in real millimetres
3. **Flatten** — the lowest 3% of vertices are clamped to one plane so it sits flat
4. **Repair** — `pymeshfix` closes marching-cubes holes and drops floating fragments
5. **Base plate** — a 3 mm plate is boolean-unioned underneath (helps multi-leg objects stand)
6. **Export** — STL (or OBJ/GLB), plus a print-readiness report

```
  1. sample_can.stl
     187.2 × 143.7 × 200.0 mm   (23,898 faces, max dim → 200 mm, merged 3 parts, pymeshfix)
     watertight: ✓ yes   boundary edges: 0   non-manifold: 0   parts: 1
     volume: 249 cm³  (~309 g PLA solid)
     overhang faces: 30%  → supports recommended
```

## Options

| Flag | Meaning |
|---|---|
| `--height MM` | scale to exactly this total height, including the base plate |
| `--size MM` | scale so the largest dimension is this (default 200) |
| `--fit-bed WxDxH` | scale to `--size` (default 200 mm), then shrink only if needed to fit, including the base plate |
| `--no-base` | skip the base plate |
| `--units m\|cm\|mm` | mesh input only: file units, kept at real size (default m; USDZ uses its own) |
| `--up y\|z` | mesh input only: up axis (default y, as ARKit exports; USDZ uses its own) |
| `--base-thickness MM` | plate thickness, default 3 |
| `--mc-resolution N` | 256 default; 320–512 = finer geometry, slower |
| `--chunk-size N` | lower if you run out of memory |
| `--texture` | bake a UV texture (see colour caveat below) |
| `--no-remove-bg` | input already has a clean background; skips rembg |
| `--foreground-ratio F` | how much of the frame the subject should fill (0.85) |
| `--format stl\|obj\|glb` | output format, default stl |
| `--device DEVICE` | torch device (default: auto mps/cuda/cpu) |
| `--keep-obj` | keep TripoSR's raw mesh next to the output |
| `--quiet` | print only output paths (progress still goes to stderr) |

## Known limits (read before blaming the tool)

- **HEIC is not supported.** Pillow has no HEIF codec here. Convert first:
  `sips -s format png IMG_1234.HEIC --out IMG_1234.png`
- **No colour in the output.** STL cannot store colour, and the repair/base steps
  rebuild topology, discarding UVs. `--texture` bakes an atlas inside TripoSR, but
  the pipeline drops it. Colour needs a GLB/3MF path that is not implemented yet.
- **The pose is whatever TripoSR decided.** A photo taken from above makes the object
  come out leaning (the network reconciles the top-down silhouette by tilting the
  object). Shoot level with the subject's mid-height for upright results. The pipeline
  does not auto-level.
- **Layer terracing.** Visible "corn rows" on curved sides are marching-cubes
  quantisation at `--mc-resolution 256`; try 320–512.
- **Not a measurement tool.** A single image yields a plausible object, not a faithful
  copy. For a real object, use many photos + photogrammetry.
- **First run downloads ~1 GB** of weights into `research/hf_cache/` and
  `research/rembg_cache/` unless `HF_HOME` / `U2NET_HOME` already point somewhere
  populated.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `can't open file '.../tools/img2mesh.py'` | Old wrapper that didn't resolve symlinks. This version does; re-check you're running this one |
| `ModuleNotFoundError: No module named 'torchmcubes'` | It has no PyPI wheel. Build it: `pip install "git+https://github.com/tatsy/torchmcubes.git"` (needs cmake + Xcode CLT) |
| `triposr: command not found` | `bin` isn't on `PATH`; check for a space in the exported path |
| `UnidentifiedImageError` | Unsupported format (HEIC/RAW). Convert to PNG/JPEG first |
| `UnavailableInvalidChannel: HTTP 403` | conda mirror refusing the host's IP (common on AutoDL). `./setup.sh --venv` |
| Hangs or OOM on batch | Each image runs in its own process on purpose; lower `--mc-resolution`/`--chunk-size` |

## Licences and attribution

- TripoSR source: MIT — see `research/triposr/LICENSE`.
- TripoSR model weights: governed by their own model card on Hugging Face
  (`stabilityai/TripoSR`); review it before commercial use.
- Background removal: `rembg` with the `bria-rmbg` ONNX model, which carries its own
  licence terms — also review before commercial use.
- If you feed this into a published model or dataset, credit the upstream projects.

## Repository notes

- `research/triposr/` holds a vendored copy of upstream TripoSR's **source only**
  (MIT). Its local checkout still contains its own `.git`, so from the repo root
  `git add research/triposr` is silently ignored (nested-repository detection).
  Add files explicitly instead, e.g. `git add research/triposr/tsr/`.
- Model weights and caches (`research/hf_cache`, `research/rembg_cache`) are
  git-ignored; `setup.sh` recreates the environment from scratch.
