# /print API

A LiDAR-scanned mesh, or a photo plus its measured height, in; a printable STL out.

Served by the same FastAPI app as the `/api/jobs` web uploader (`server/app.py`); the
route lives in `server/print_api.py`. The Modal deploy exposes only `/print` and
`/api/health`.

```bash
python -m uvicorn server.app:app --port 6006   # local (CPU works, photos are slow)
modal deploy deploy/modal_app.py               # serverless GPU, see that file
```

## `POST /print`

`multipart/form-data`, exactly one of `mesh` or `photo`.

| Field | Type | Default | Meaning |
|---|---|---|---|
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
| `camera_transform` | JSON, optional | 16 floats, column-major camera-to-world (ARKit, world +Y up): gravity for the floor and upright, also supported for photo-only requests without depth |
| `image_orientation` | 1 \| 3 \| 6 \| 8 | EXIF value; upright fallback without `camera_transform` |
| `object_points` | JSON, optional | `[[u, v], ...]` taps on the object in photo pixels; picks the object region |

Steps: mask the object (rembg, taps), back-project its depth pixels, find the floor (gravity
plus a RANSAC plane), align the mesh by trimmed similarity ICP from 8 yaw starts, pull the
seen vertices onto the depth surface and fade that into the hidden side, then the usual
flatten / repair / base plate. Too few depth points on the object (< 200) falls back to the
photo + `height_mm` path, reported as `"fusion": {"skipped": "..."}`.

Photo meshes that were not depth-fused are leveled before scaling and adding the
plate. Without a camera transform, area-weighted face normals are clustered
within 10°, retaining groups covering at least 2.5% of the surface. Candidate
up directions within 35° of +Z are scored by parallel and perpendicular plane
area, then refined by weighted least squares. A correction must clearly beat
the current +Z score (by both 10% of that score and 5% of total area); otherwise
the mesh is unchanged. This supports box-like objects with missing or flattened
bottoms. If that estimate is inconclusive, a pitch-only search rotates about
TripoSR's camera-horizontal +Y axis in the downward-camera correction direction
(negative Y rotation), from 0–75° in 1° steps. It minimises AABB volume after
optimising yaw in 1° steps over 0–89°; yaw is used only for scoring, not applied.
The pitch correction must reduce that volume by more than 5% versus pitch zero.
Both geometric estimators report source `shape`; neither can repair warped
geometry or guarantee that a semantic top is facing up.
With a transform (no depth required), camera gravity takes
precedence, including the same sensor-image upright quarter turns as depth fusion;
`image_orientation` is validated but does not override supplied gravity.
`X-Print-Report` includes `"level": {"source": "gravity" | "shape" | "none",
"tilt_deg": 19.8}` for these photo meshes. The angle is the applied correction.
Depth-fused meshes and uploaded meshes retain their existing behavior.

For photo-only requests with `camera_transform`, the iOS app uploads an already
upright photo with `image_orientation=1`. The column-major matrix must be the
camera-to-world of that **upright photo camera** (inverse of ARKit's
`viewMatrix(for: interfaceOrientation)`): camera +X is pixel-right, +Y is
pixel-up, viewing direction is -Z, and world +Y is up. The server derives upright
turns from that matrix; an upright camera pitched down (including 50°) gets zero
quarter turns, so an EXIF-free upright photo is not rotated again. Depth requests
retain the sensor-oriented photo plus `ARCamera.transform` contract above.

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

## `GET /api/health`

`{"status": "ok"}` (more fields locally), no auth.

## Examples

```bash
curl -F mesh=@scan.usdz -H "Authorization: Bearer $TOKEN" $URL/print -o scan.stl
curl -F photo=@mug.jpg -F height_mm=95 -H "Authorization: Bearer $TOKEN" $URL/print -o mug.stl
```

Timing: a mesh takes a few seconds; a photo about 10–20 s on a warm GPU, plus
~30–60 s when the container is cold (it scales to zero after 5 idle minutes).
