#!/bin/bash
# setup.sh — build the Python environment for this pipeline and verify it.
#
#   ./setup.sh              auto: try conda, fall back to a repo-local .venv
#   ./setup.sh --venv       force a plain venv (no conda needed)
#   ./setup.sh --conda      force the conda env
#   ./setup.sh --link       also symlink bin/triposr into ~/.local/bin
#   ./setup.sh --dry-run    print the commands without running them
#
# Environment knobs:
#   ENV_NAME        conda env name                   (default: triposr)
#   PYTHON_BIN      base interpreter for venv mode   (default: python3.11, then python3)
#   PIP_INDEX_URL   package mirror, e.g. for China:
#                   https://pypi.tuna.tsinghua.edu.cn/simple
#   GITHUB_MIRROR   prefix for git clones when github.com is blocked,
#                   e.g. https://ghproxy.com/
#
# Why a venv fallback exists: on shared GPU hosts (AutoDL, Colab-like images) the
# preconfigured conda channels are often mirrors that answer 403 for the machine's
# IP — e.g. "UnavailableInvalidChannel: HTTP 403 FORBIDDEN for channel
# anaconda/pkgs/main". That failure is unrelated to this project, and a venv built
# from the image's own Python sidesteps it entirely.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-triposr}"
MODE="auto"
LINK=0
DRY=0

for arg in "$@"; do
    case "$arg" in
        --venv)    MODE="venv" ;;
        --conda)   MODE="conda" ;;
        --link)    LINK=1 ;;
        --dry-run) DRY=1 ;;
        -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown option $arg" >&2; exit 2 ;;
    esac
done

run() {
    if [ "$DRY" = "1" ]; then
        echo "    [dry-run] $*"
    else
        "$@"
    fi
}

pip_install() {  # pip_install <python> <args...>
    local py="$1"; shift
    if [ -n "${PIP_INDEX_URL:-}" ]; then
        run "$py" -m pip install -i "$PIP_INDEX_URL" "$@"
    else
        run "$py" -m pip install "$@"
    fi
}

pick_python() {
    local candidate
    for candidate in "${PYTHON_BIN:-}" python3.11 python3; do
        [ -z "$candidate" ] && continue
        command -v "$candidate" >/dev/null 2>&1 || continue
        local version
        version="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
        if [ "$version" != "3.11" ]; then
            echo "warning: $candidate is Python $version; the pinned wheels were tested on 3.11" >&2
        fi
        echo "$candidate"
        return 0
    done
    return 1
}

install_into() {  # install_into <python> <label>
    local py="$1" label="$2"
    echo "==> installing dependencies into $label"
    pip_install "$py" --upgrade pip setuptools wheel >/dev/null 2>&1 || true
    pip_install "$py" -r "$HERE/research/triposr_requirements.txt"

    echo "==> building torchmcubes (no PyPI wheel exists)"
    if [ "$DRY" != "1" ] && "$py" -c "import torchmcubes" 2>/dev/null; then
        echo "    already installed; skipping"
    else
        local url="https://github.com/tatsy/torchmcubes.git"
        local mirror="${GITHUB_MIRROR:-}"
        if [ -n "$mirror" ]; then
            pip_install "$py" "git+${mirror}${url}" --no-build-isolation || \
                pip_install "$py" "git+${url}" --no-build-isolation
        else
            pip_install "$py" "git+${url}" --no-build-isolation || {
                echo "    direct clone failed; retrying via https://ghproxy.com/" >&2
                pip_install "$py" "git+https://ghproxy.com/${url}" --no-build-isolation
            }
        fi
    fi

    echo "==> verifying imports"
    if [ "$DRY" = "1" ]; then
        echo "    [dry-run] $py -c 'import torch, torchmcubes, trimesh, pymeshfix, rembg'"
    else
        "$py" - <<'PYEOF'
import torch, torchmcubes, trimesh, pymeshfix, rembg, transformers
print(f"    torch {torch.__version__}  cuda={torch.cuda.is_available()}"
      f"  mps={torch.backends.mps.is_available()}")
print(f"    trimesh {trimesh.__version__}  rembg {rembg.__version__}")
print("    all imports OK")
PYEOF
    fi
}

setup_venv() {
    local base
    base="$(pick_python)" || {
        echo "error: no python3 found" >&2
        echo "       install Python 3.11, or set PYTHON_BIN=/path/to/python3.11" >&2
        exit 1
    }
    echo "==> venv mode using $base ($("$base" -V 2>&1))"
    if [ ! -x "$HERE/.venv/bin/python" ]; then
        run "$base" -m venv "$HERE/.venv"
    else
        echo "    .venv already exists; reusing"
    fi
    install_into "$HERE/.venv/bin/python" "$HERE/.venv"
}

setup_conda() {
    local conda="${CONDA_EXE:-$(command -v conda || true)}"
    if [ -z "$conda" ]; then
        return 1
    fi
    echo "==> conda: $conda"

    if "$conda" env list 2>/dev/null | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        echo "==> env '$ENV_NAME' already exists; reusing"
    else
        echo "==> creating env '$ENV_NAME' (python 3.11)"
        if ! run "$conda" create -n "$ENV_NAME" python=3.11 -y; then
            cat >&2 <<'MSG'

    conda could not reach its channels. On shared/GPU hosts this is usually a
    mirror refusing the machine's IP, e.g.:

        UnavailableInvalidChannel: HTTP 403 FORBIDDEN for channel anaconda/pkgs/main
        <https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main>

    That is a network/mirror problem, not this project's. Options:

      1. Skip conda entirely (recommended on AutoDL and similar):
             ./setup.sh --venv
      2. Use a reachable channel:
             conda create -n triposr python=3.11 -y -c conda-forge --override-channels
      3. Point conda at a mirror that answers 200 for you:
             conda config --show-sources        # see what is configured
             conda config --remove-key default_channels   # then let conda use defaults

MSG
            return 1
        fi
    fi
    local py
    if [ "$DRY" = "1" ]; then
        echo "    [dry-run] $conda run -n $ENV_NAME python -c 'import sys; print(sys.executable)'"
        py="$ENV_NAME/bin/python"
    else
        py="$("$conda" run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)' | tail -1)"
    fi
    echo "==> interpreter: $py"
    install_into "$py" "conda env '$ENV_NAME'"
}

chmod +x "$HERE/bin/triposr" "$HERE/tools/img2mesh.py" 2>/dev/null || true

case "$MODE" in
    venv)
        setup_venv ;;
    conda)
        setup_conda || exit 1 ;;
    auto)
        if ! setup_conda; then
            echo
            echo "==> conda path failed; falling back to a repo-local .venv"
            setup_venv
        fi ;;
esac

if [ "$LINK" = "1" ]; then
    mkdir -p "$HOME/.local/bin"
    ln -sf "$HERE/bin/triposr" "$HOME/.local/bin/triposr"
    echo "==> linked: $HOME/.local/bin/triposr"
fi

echo
echo "Done. Try it:"
echo "    $HERE/bin/triposr $HERE/examples/sample_can.jpg"
echo
echo "First run downloads model weights (HF_HOME cache, ~1 GB) unless already present."
echo "Behind a firewall, mirror them: HF_ENDPOINT=https://hf-mirror.com"
echo "Web app:  $HERE/.venv/bin/python -m uvicorn server.app:app --host 0.0.0.0 --port 6006"
