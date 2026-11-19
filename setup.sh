#!/bin/bash
# setup.sh — build the `triposr` conda env and verify it.
#
#   ./setup.sh            create/refresh the env, then smoke-test imports
#   ./setup.sh --link      also symlink bin/triposr into ~/.local/bin
#
# Safe to re-run: an existing env is reused, not destroyed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-triposr}"
CONDA="${CONDA_EXE:-$(command -v conda || true)}"
LINK=0
[ "${1:-}" = "--link" ] && LINK=1

if [ -z "$CONDA" ]; then
    echo "error: conda not found on PATH." >&2
    echo "       install Miniconda, or set CONDA_EXE=/path/to/conda" >&2
    exit 1
fi
echo "==> conda: $CONDA"

if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "==> env '$ENV_NAME' already exists; reusing"
else
    echo "==> creating env '$ENV_NAME' (python 3.11)"
    "$CONDA" create -n "$ENV_NAME" python=3.11 -y
fi

PY="$("$CONDA" run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)' | tail -1)"
echo "==> interpreter: $PY"

echo "==> installing pinned requirements"
"$PY" -m pip install --upgrade pip setuptools wheel -q
"$PY" -m pip install -r "$HERE/research/triposr_requirements.txt"

echo "==> building torchmcubes from source (no PyPI wheel exists)"
if "$PY" -c "import torchmcubes" 2>/dev/null; then
    echo "    already installed; skipping"
else
    "$PY" -m pip install "git+https://github.com/tatsy/torchmcubes.git" --no-build-isolation
fi

echo "==> verifying imports"
"$PY" - <<'PYEOF'
import torch, torchmcubes, trimesh, pymeshfix, rembg, transformers
print(f"    torch {torch.__version__}  mps={torch.backends.mps.is_available()}"
      f"  cuda={torch.cuda.is_available()}")
print(f"    trimesh {trimesh.__version__}  rembg {rembg.__version__}")
print("    all imports OK")
PYEOF

chmod +x "$HERE/bin/triposr" "$HERE/tools/img2mesh.py"

if [ "$LINK" = "1" ]; then
    mkdir -p "$HOME/.local/bin"
    ln -sf "$HERE/bin/triposr" "$HOME/.local/bin/triposr"
    echo "==> linked: $HOME/.local/bin/triposr"
fi

echo
echo "Done. Try it:"
echo "    $HERE/bin/triposr $HERE/examples/sample_can.jpg"
echo
echo "First run downloads ~1 GB of model weights (cached in research/hf_cache)."
