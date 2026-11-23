# Print server on our own box

`/print`, `/print/uploads`, `/photo-check` and `/api/health` run here: auth, uploads and
S3 signing, job state and NDJSON streams, the SiliconFlow photo check, depth fusion,
repair, printability, the STL upload to S3 and slicing every STL into G-code for the CUHK
MakerSpace Creality K1C (PrusaSlicer CLI, installed by `deploy.sh` under `/opt`). Only TripoSR + rembg inference runs on
Modal (`deploy/modal_triposr.py`), called through `server/remote_triposr.py`
(`TRIPOSR_BACKEND=modal`).

- Host: `root@47.237.92.84` (SSH key login), URL `https://47-237-92-84.sslip.io`
- Caddy terminates HTTPS (Let's Encrypt) and proxies to uvicorn on `127.0.0.1:8000`
  without buffering, so NDJSON streams flow through.
- Job state is in memory in one uvicorn process; a restart fails in-flight jobs.

## Secrets

`/etc/3d-print/env`, mode 600, one `NAME=value` per line, never in git:

| Name | Use |
|---|---|
| `PRINT_API_TOKEN` | bearer token the app sends |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `PRINT_S3_BUCKET` | signed uploads and STL results |
| `SILICONFLOW_API_KEY` | `/photo-check` VLM |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | calling TripoSR on Modal |

## Deploy

```bash
modal deploy deploy/modal_triposr.py       # only when the inference side changes
ssh root@47.237.92.84 'curl -fsSL https://raw.githubusercontent.com/1155194577/3d-print/main/deploy/box/deploy.sh | bash -s -- main'
```

`deploy.sh` checks out the ref in `/opt/3d-print/app`, installs `deploy/box/requirements.txt`
into `/opt/3d-print/venv`, installs the systemd unit and Caddyfile, and restarts.
Logs: `journalctl -u 3d-print -f`.
