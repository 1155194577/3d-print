# /print API

A LiDAR-scanned mesh, or a photo plus its measured height, in; a printable STL out.

Served by the same FastAPI app as the `/api/jobs` web uploader (`server/app.py`); the
route lives in `server/print_api.py`. The Modal deploy exposes only `/print`,
`/photo-check` and `/api/health`.

```bash
python -m uvicorn server.app:app --port 6006   # local (CPU works, photos are slow)
modal deploy deploy/modal_app.py               # serverless GPU, see that file
```

## `POST /print`

`multipart/form-data`, exactly one of `mesh` or `photo`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `mode` | `object` \| `face` | `object` | Face mode requires a photo and produces a flat-cut bust. |
| `face_transform` | JSON array of 16 floats | | Optional column-major `ARFaceAnchor.transform` in ARKit world coordinates (origin at head centre, +Y face up). Used only in face mode; requires `camera_transform` for depth fusion. |
| `mesh` | file | | LiDAR scan: `.usdz`, `.obj`, `.ply` (also `.glb`, `.stl`). Skips TripoSR. |
| `photo` | file | | `.jpg`, `.png` or `.webp` (not HEIC). EXIF rotation is honoured. |
| `height_mm` | number > 0 | | Measured height of the object in mm. Photo: replaces the 200 mm default. Mesh: overrides the scan's real size. |
| `units` | `m` \| `cm` \| `mm` | `m` | Units of an OBJ/PLY/GLB/STL mesh. USDZ uses its own `metersPerUnit`. |
| `up_axis` | `y` \| `z` | `y` | Up axis of an OBJ/PLY/GLB/STL mesh (ARKit exports Y-up). USDZ uses its own. |
| `fit_bed` | `WxDxH` mm | | Shrink, only if needed, to fit this printer bed, e.g. `220x220x250`. |
| `base_plate` | bool | `true` | Fuse a 3 mm plate underneath (adds 2.5 mm of height and a margin around). |

Header `Authorization: Bearer <token>` is required when the server has `PRINT_API_TOKEN` set (the Modal deploy does).

Sizing: a mesh keeps its real size (metres → mm). `height_mm` is the object itself; the
base plate adds 2.5 mm below it, so a 120 mm object comes back 122.5 mm tall. A photo
without `height_mm` is scaled to 200 mm on its largest side.

### Photo + LiDAR depth

Add these to a `photo` request, all from the same ARFrame, and the server fits TripoSR's
shape to the measured depth. True size and pose come from the depth (so `height_mm` is
only a fallback), the seen side is the LiDAR surface, and TripoSR fills in the hidden side.

| Field | Type | Meaning |
|---|---|---|
| `photo` | file | `capturedImage` in sensor orientation (not rotated), so `intrinsics` apply as-is |
| `depth` | file `.f32` | `sceneDepth.depthMap`: little-endian float32 metres, row-major, NaN = no data (or a 16-bit `.png` in mm, 0 = no data) |
| `depth_width`, `depth_height` | int | depth map size, e.g. 256 × 192 (raw depth only) |
| `depth_confidence` | file `.u8`, optional | `confidenceMap`, uint8 0/1/2, same size; 0 is dropped |
| `intrinsics` | JSON | `{"fx","fy","cx","cy","width","height"}` in photo pixels; scaled to the depth size |
| `depth_intrinsics` | JSON, optional | Same keys, calibrated depth camera values in depth pixels. Width/height describe that calibration; scaled to the uploaded depth map if different. Overrides the depth K derived from `intrinsics`. |
| `depth_extrinsics` | JSON, optional | 16 finite floats, column-major depth-camera → photo-camera transform in metres, OpenCV axes (x right, y down, z forward). Default identity. |
| `camera_transform` | JSON, optional | 16 floats, column-major camera-to-world (ARKit, world +Y up): gravity for the floor and upright |
| `image_orientation` | 1 \| 3 \| 6 \| 8 | EXIF value; upright fallback without `camera_transform` |
| `object_points` | JSON, optional | `[[u, v], ...]` taps on the object in photo pixels; picks the object region |

Steps: mask the object (rembg, taps), back-project its depth pixels, find the floor (gravity
plus a RANSAC plane), align the mesh by trimmed similarity ICP from 8 yaw starts, pull the
seen vertices onto the depth surface and fade that into the hidden side, then the usual
flatten / repair / base plate. Too few depth points on the object (< 200) falls back to the
photo + `height_mm` path, reported as `"fusion": {"skipped": "..."}`.

For differently calibrated cameras or aspect ratios (e.g. 640 × 480 depth and
1280 × 720 photo), supply `depth_intrinsics` as well as the photo `intrinsics`.
The server back-projects each depth pixel, applies `depth_extrinsics`, and projects
into the photo to sample its foreground mask and register object taps. Gravity
is rotated into the depth camera frame. Without `depth_intrinsics`, the original
mask resize / scaled photo intrinsics path is retained. These fields apply to
both object and face modes.

### Face busts (`mode=face`)

Send a `photo`, optionally with front-camera TrueDepth from the same ARFrame.
On iOS, encode `ARFrame.capturedImage` in sensor orientation, **not mirrored**;
convert `ARFrame.capturedDepthData` to `DepthFloat32` and upload its depth map as
row-major little-endian float32 metres (e.g. 640 × 480), without a confidence map.
Send `ARCamera.intrinsics` with the **capturedImage** width and height, and
`ARCamera.transform` as `camera_transform`. The optional `face_transform` is
`ARFaceAnchor.transform`, also column-major (16 finite floats).

Without depth, send `image_orientation` (1, 3, 6 or 8) to turn a sensor-orientation
JPEG upright before reconstruction, even when it has no EXIF tag. This also works
in object mode. If omitted, photo-only requests retain EXIF-based orientation;
an explicit 1 leaves the image as stored. Optional `face_vertices` (float32 LE XYZ)
and `face_vertex_count` capture metadata are accepted and ignored, whether sent as
file parts or form fields.

Face fusion uses gravity, not a floor plane, rejects points more than 25 cm behind
the nearest foreground point, and cuts 12 cm below the face anchor's head centre.
Without an anchor it cuts 24 cm below the foreground top, clamped to its lowest
point. The cap makes the bust stand flat. The fusion report adds `cut`:
`"face anchor"` or `"top - 24 cm"`.

Face mode always scales the object to **100 mm tall** by default, including after
depth fusion; `height_mm` overrides this, and the base plate adds 2.5 mm. A photo
alone (or unusable depth) uses TripoSR and caps off the bottom 20% of its height;
unusable depth is reported under `fusion.skipped` as above. `face_transform` is
ignored in object mode; malformed face-mode transforms return 400. Mesh uploads
are not accepted in face mode. `X-Print-Report` includes `"mode": "face"`.

### Response

- `200`, `Content-Type: model/stl`, binary STL, `Content-Disposition: attachment; filename="<upload name>.stl"`.
- `X-Print-Report` header, JSON:
  `{"extents_mm": [x, y, z], "watertight": true, "faces": 23898, "volume_cm3": 249.0, "scale": "height → 122.5 mm"}`.
  With depth it adds `"fusion": {"lidar_points": 2380, "icp_rmse_mm": 0.8, "scale": 0.37, "seen_vertices": 0.32, "floor": "camera gravity + plane"}`
- Every STL is checked for printing (`tools/printability.py`) and the report adds `"printability"`:
  `{"printable": true, "errors": [], "warnings": ["38% of the surface overhangs more than 45°: print with supports"], "open_edges": 0, "nonmanifold_edges": 0, "parts": 1, "base_contact_mm2": 1810.0, "overhang_pct": 38.2, "thin_wall_pct": 0.4, "min_wall_mm": 1.9, "solid_mass_g": 309.0}`.
  Warnings (overhangs, walls under 0.8 mm, floating parts, a base that may tip, bigger than a 220×220×250 bed) never block the STL; show them to the user.
  Holes, bad winding and floating parts get one repair pass first, listed in `"repaired"`.
  Only an STL that is still not watertight after that, or doesn't fit `fit_bed`, is refused with 422.

Errors are JSON `{"detail": "..."}`:

| Code | When |
|---|---|
| 400 | neither or both files, empty file, bad `fit_bed`, `depth` without `photo`, malformed depth/intrinsics/transform |
| 401 | missing or wrong bearer token |
| 415 | unsupported file extension (e.g. `.heic`, `.fbx`) |
| 413 | upload over 200 MB (`PRINT_MAX_UPLOAD_MB`) |
| 422 | invalid field value, a file that does not parse or reconstruct into a mesh, or an STL that can't be printed (still not watertight after repair, or doesn't fit `fit_bed`) |

### Streamed progress and print jobs

Send `Accept: application/x-ndjson` and the print becomes a job that runs on the server whether
or not the app stays connected. Auth and validation errors (400/401/413/415) are still plain
JSON errors. Otherwise the response is `200 application/x-ndjson`, one JSON object per line,
flushed as it goes:

```
{"type":"accepted","job_id":"5f0c…"}
{"type":"progress","stage":"received","progress":0.02,"message":"Upload received","elapsed_s":0.0}
{"type":"progress","stage":"reconstruct","progress":0.21,"message":"Reconstructing the shape","elapsed_s":6.1}
...
{"type":"done","progress":1.0,"filename":"mug.stl","report":{...X-Print-Report...},"stl_base64":"...","elapsed_s":41.8}
```

- Stages, in order: `received`, `background`, `reconstruct`, `mesh`, `fuse` (depth only), `repair`, `export`, `check`. A mesh goes straight from `received` to `repair`. Each stage is sent once, in order.
- The current stage repeats every 2 s as a heartbeat. Its `progress` creeps toward the next stage's start and never reaches it.
- The last line is exactly one `done`, or `{"type":"error","status":422,"detail":"...","elapsed_s":...}`. A stream that ends without either means the connection dropped, not that the print failed: resume it.
- Optional form field `job_id` (a UUID the app picks). A retry with the same id joins that job instead of starting a second print.

Resuming, from any container, for 24 h after the job starts (unknown or expired ids are 404):

| Route | Answer |
|---|---|
| `GET /print/{job_id}` + `Accept: application/x-ndjson` | the latest line, then the stream as above to `done` (without `stl_base64`) or `error` |
| `GET /print/{job_id}` | the latest line as one JSON object |
| `GET /print/{job_id}/result` | held open until the job ends: `model/stl` + `X-Print-Report`, or the job's error as `{"detail"}` with its status |

On Modal, job state lives in the `3d-print-jobs` modal.Dict, so any container can answer. A job whose
container dies (a redeploy) turns into an error within 90 s. Nothing arrives while Modal cold-starts
the GPU container (up to about 4 minutes).

## `POST /photo-check`

Run it on a Photo Print shot right after capture, before `/print`. A vision-language
model (Qwen2.5-VL-3B, `server/photo_check.py`) looks for clutter and a cut-off object, and
the phone's measured tilt catches a steep angle, so the app can ask for a retake and still
offer to print anyway. Same bearer token as `/print`.

`multipart/form-data`:

| Field | Type | Meaning |
|---|---|---|
| `photo` | file | The shot: `.jpg`, `.png` or `.webp`. EXIF rotation is honoured. |
| `camera_transform` | JSON, 16 floats | ARKit `camera.transform` at capture (column-major camera-to-world, +Y up), as sent to `/print`. Its pitch decides `steep_angle`; without it the angle is not checked (the model can't judge it: Qwen2.5-VL 3B and 7B both called a 54° top-down shot "eye level"). |

`200`, JSON:

```json
{
  "ok": false,
  "issues": [
    {"code": "steep_angle", "message": "The camera is pointing down at the object. Hold the phone level with its middle and shoot it from the side."},
    {"code": "cluttered_background", "message": "There are other things around the object. Move it to a clear spot or clear the background."}
  ],
  "angle_checked": true,
  "camera_pitch_deg": 54.0,
  "max_pitch_deg": 30.0,
  "model": "Qwen/Qwen2.5-VL-3B-Instruct",
  "answers": {"cluttered_background": "Yes.", "cut_off": "Yes."}
}
```

- `ok`: `true` when `issues` is empty; print straight away.
- `issues[].code`, in this order: `steep_angle` (camera tilted more than `max_pitch_deg`, default 30, below level), `cluttered_background` (other things around or behind the object), `cut_off` (part of the object outside the frame). Codes are stable; show `message` to the user.
- `angle_checked` / `camera_pitch_deg`: whether `camera_transform` was sent, and its pitch (0 level, 90 straight down).
- `answers`: the model's raw replies to its two yes/no questions (`cut_off` asks "is the whole object inside?"), for debugging.

Errors are JSON `{"detail": "..."}`: 400 empty photo or malformed `camera_transform`, 401 token,
413 over 25 MB, 415 extension, 422 unreadable image, 502 the model failed. A failed
check should not block printing: the app can fall through to `/print`.

Timing: about 1 s on a warm GPU; the model has its own container, so add ~30–60 s
when it is cold.

## `GET /api/health`

`{"status": "ok"}` (more fields locally), no auth.

## Examples

```bash
curl -F mesh=@scan.usdz -H "Authorization: Bearer $TOKEN" $URL/print -o scan.stl
curl -F photo=@mug.jpg -F height_mm=95 -H "Authorization: Bearer $TOKEN" $URL/print -o mug.stl
curl -F photo=@mug.jpg -H "Authorization: Bearer $TOKEN" $URL/photo-check
```

Timing: a mesh takes a few seconds; a photo about 10–20 s on a warm GPU, plus
~30–60 s when the container is cold (it scales to zero after 5 idle minutes).

## Modal deployment and terminated requests

Deploy with `modal deploy --strategy rolling deploy/modal_app.py`, then wait for
`GET /api/health` before sending a print. Do not stop containers while they are
starting or serving requests to force a new deployment: stopping a container also
kills its in-flight requests, including requests still waiting for model loading.
A client can then receive `500 modal-http: internal error: function was terminated
by signal` without a Python traceback. Use Modal's container result/termination
reason to distinguish an operator stop from an OOM or native crash; the HTTP
message alone does not identify the cause.

Inspect `modal app history 3d-print --json` and
`modal container logs <container-id> --all --timestamps` when correlating a failure
with deployments or stops. A result of `GENERIC_STATUS_TERMINATED` with
`Container terminated due to user termination request` is an explicit stop, not
evidence that face cutting crashed.

To isolate native face processing in the **same image** as the deployment:

```bash
modal run -m deploy.diagnose_face --photo face.jpg --mesh raw-face.obj
modal run -m deploy.diagnose_face --photo face.jpg --mesh raw-face.obj --operation cut
modal run -m deploy.diagnose_face --photo face.jpg --mesh raw-face.obj --operation object
```

The first run reconstructs the photo on an L4 and saves the raw mesh locally;
subsequent runs reuse it. A subprocess in the deployment image prints package
versions, mesh counts, peak RSS, and cut/repair stage boundaries with faulthandler
enabled. A native failure produces a nonzero exit code instead of silently
finishing the diagnostic. `object` skips the bust cut as a control. Keep the
photo and raw mesh out of git.
