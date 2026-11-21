#!/usr/bin/env bash
# Deploy or update the print server on the box. Run as root on the box:
#   curl -fsSL https://raw.githubusercontent.com/1155194577/3d-print/main/deploy/box/deploy.sh | bash -s -- [git ref]
# Secrets live in /etc/3d-print/env (mode 600), see deploy/box/README.md.
set -euo pipefail

REF=${1:-main}
REPO=${REPO:-https://github.com/1155194577/3d-print.git}
ROOT=/opt/3d-print
export PATH="$HOME/.local/bin:$PATH"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v caddy >/dev/null || { apt-get update -qq && apt-get install -y -qq caddy; }

mkdir -p "$ROOT"
[ -d "$ROOT/app/.git" ] || git clone -q "$REPO" "$ROOT/app"
git -C "$ROOT/app" fetch -q origin "$REF"
git -C "$ROOT/app" checkout -q --detach FETCH_HEAD

[ -x "$ROOT/venv/bin/python" ] || uv venv -q -p 3.11 "$ROOT/venv"
uv pip install -q --python "$ROOT/venv/bin/python" \
    -r "$ROOT/app/deploy/box/requirements.txt" -c "$ROOT/app/research/triposr_requirements.txt"

if [ ! -f /etc/3d-print/env ]; then
    echo "missing /etc/3d-print/env (see deploy/box/README.md)" >&2
    exit 1
fi
install -m 644 "$ROOT/app/deploy/box/3d-print.service" /etc/systemd/system/3d-print.service
install -m 644 "$ROOT/app/deploy/box/Caddyfile" /etc/caddy/Caddyfile
systemctl daemon-reload
systemctl enable -q 3d-print caddy
systemctl restart 3d-print
systemctl reload caddy || systemctl restart caddy

for _ in $(seq 30); do
    curl -fs http://127.0.0.1:8000/api/health >/dev/null && break
    sleep 1
done
curl -fs http://127.0.0.1:8000/api/health
echo " deployed $(git -C "$ROOT/app" rev-parse --short HEAD)"
