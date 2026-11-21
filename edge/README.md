# TripoSR on the iPhone (Core ML, quantized)

TripoSR split into two Core ML models so the native iOS app
([lidar-scanner-ios](https://github.com/1155194577/lidar-scanner-ios), **Settings → Offline Photo Print**)
can turn a photo into a mesh with no internet:

| model | inputs | output |
|---|---|---|
| `TripoSREncoder` | `image`: 512×512 RGB, object centred at 85 % on 50 % grey | `triplanes` 3×40×64×64 |
| `TripoSRDecoder` | `triplanes`, `points` 65536×3 in [-0.87, 0.87] | `density_logit` 65536×1 |

density = exp(logit); the surface is density = 25 (TripoSR's default), found with marching cubes on a
grid of decoder queries (x-major, as `grid_points` builds it). The decoder returns the logit because
exp overflows float16, which Core ML uses on the GPU and Neural Engine.

## Download

[Release `triposr-coreml-v1`](https://github.com/1155194577/3d-print/releases/tag/triposr-coreml-v1)
holds both models in both quantizations, each `.mlpackage` as three files
(`<Model>-<q>-Manifest.json`, `-model.mlmodel`, `-weight.bin`) that go back together as
`<Model>.mlpackage/{Manifest.json, Data/com.apple.CoreML/model.mlmodel, Data/com.apple.CoreML/weights/weight.bin}`.

| quantization | encoder | decoder | needs |
|---|---|---|---|
| `int8` (what the app downloads) | 411 MB | 0.1 MB | iOS 17 |
| `int4-block32` | 240 MB | 0.1 MB | iOS 18 |

The float32 checkpoint is 1.6 GB. Encoder linear and conv weights are compressed; the decoder MLP
(45 K weights) stays float16. Activations run in float16.

## Quality

`evaluate.py` runs the same split in PyTorch with the weights rounded the way coremltools rounds them
and compares each scheme with float32 at marching-cubes resolution 256: triplane relative error, IoU of the
occupied grid, and Chamfer distance between the meshes as a % of the float32 mesh's bounding-box diagonal.

Six images (`chair`, `flamingo`, `hamburger`, `robot`, `sample_can`, `teapot`), float16 activations as on
the phone measured on `chair` only (they add < 0.003 IoU to the weight error):

| weights | triplane error | occupancy IoU, mean (worst) | Chamfer % diag, mean (worst) | watertight |
|---|---|---|---|---|
| fp16 | 0.0002 | 0.9998 (0.9997) | 0.15 (0.25) | 6/6 |
| **int8** | 0.0089 | **0.9932 (0.9844)** | **0.25 (0.30)** | 6/6 |
| int4-block32 | 0.0800 | 0.9234 (0.8751) | 0.39 (0.58) | 6/6 |

int8 is indistinguishable from float32 at print scale (0.3 % of a 100 mm print is 0.3 mm) and is what
the app ships; int4 saves 170 MB but loses thin parts (worst IoU 0.875 on the can) and needs iOS 18.

## Rebuild

```sh
python -m venv ~/qenv && ~/qenv/bin/pip install -r research/triposr_requirements.txt coremltools==9.0 scikit-image pytest
# config.yaml + model.ckpt from https://huggingface.co/stabilityai/TripoSR
~/qenv/bin/python edge/export_coreml.py --model-dir ~/models --quantization int8 --out build/coreml/int8
~/qenv/bin/python edge/export_coreml.py --model-dir ~/models --quantization int4-block32 --out build/coreml/int4
~/qenv/bin/python edge/evaluate.py --model-dir ~/models --out eval.json examples/*.png
TRIPOSR_MODEL_DIR=~/models ~/qenv/bin/python -m pytest edge
```

Exporting works on Linux; running the `.mlpackage` needs macOS or iOS.
