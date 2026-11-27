# RGB-only video reconstruction benchmark

Research benchmark for a possible no-LiDAR mode. It compares released video reconstruction models on [TUM RGB-D Freiburg 1 desk](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download). Models see **RGB only**. TUM depth and motion-capture poses are used only as reference. This does not run the scanner's iPhone LiDAR or TripoSR pipelines: TUM's RGB-D reference is not iPhone LiDAR, and TripoSR reconstructs an object from a single photo, not scene geometry and a camera trajectory from a clip.

## Results

One 24-frame, 19.84 s clip. Lower is better except δ < 1.25 (higher is better).

| Model | Mode | ATE RMSE (m) | RPE trans. RMSE (m) | RPE rot. RMSE (°) | Depth AbsRel | δ < 1.25 | Accuracy (m) | Completeness (m) | Chamfer-L1 (m) | CPU wall (s) | Peak RSS (GiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LingBot-Map 2.0 | Causal, CPU SDPA port | 0.0316 | 0.0366 | 1.375 | 0.0300 | 98.16% | 0.0405 | 0.0333 | 0.0369 | **228.43** | 10.16 |
| LingBot-Map 1.0 | Causal, upstream SDPA | 0.0344 | 0.0437 | 1.575 | 0.0312 | 98.40% | **0.0374** | **0.0276** | **0.0325** | 273.77 | 10.35 |
| VGGT-1B | Offline | 0.0235 | 0.0339 | 1.452 | 0.0279 | 97.99% | 0.0411 | 0.0369 | 0.0390 | 340.99 | 8.99 |
| π3 | Offline | **0.0175** | **0.0267** | **1.281** | **0.0258** | **98.64%** | 0.0497 | 0.0490 | 0.0494 | 285.19 | **7.49** |

LingBot-Map 2.0 was the fastest CPU run and improved trajectory error over 1.0, but 1.0 produced the closest fused cloud, while π3 had the best pose and scale-aligned depth scores. That makes 2.0 worth a follow-up streaming test on actual phone footage, not a demonstrated replacement for the scanner's current modes. Carry 1.0 as the streaming baseline and π3 as the offline quality baseline.

## Setup and reproduction

Run from this directory. Reference environment: Linux x86-64, Python 3.10, 8 CPU threads, PyTorch 2.8.0 CPU. The downloads total about 18 GB.

```sh
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install 'torch==2.8.0+cpu' 'torchvision==0.23.0+cpu' \
  'numpy==1.26.4' 'opencv-python-headless==4.11.0.86' 'scipy==1.15.3' \
  'matplotlib==3.10.9' 'pillow==12.3.0' 'einops==0.8.2' 'safetensors==0.8.0' \
  'huggingface_hub==2.5.0' 'tqdm==4.70.1' \
  --extra-index-url https://download.pytorch.org/whl/cpu

mkdir -p data weights outputs
curl -fL --retry 3 https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz -o data/tum.tgz
tar -xzf data/tum.tgz -C data

clone_at() {
  git clone "$1" "data/$2"
  git -C "data/$2" checkout --detach "$3"
}
clone_at https://github.com/Robbyant/lingbot-map-v2 lingbot-map-v2 1b896cdfd4043ddab4b414df1de7f0915691a6cb
clone_at https://github.com/Robbyant/lingbot-map lingbot-map 62234bfd0c5fb0ea6e5b2b12910d3873e342a7e8
clone_at https://github.com/facebookresearch/vggt vggt a288dd0f14786c93483e45524328726ab7b1b4ce
clone_at https://github.com/yyfz/Pi3 Pi3 a65e87ee9fb84ee4fd7a15ff8d8a26d16119da99

hf_file() {
  curl -fL --retry 3 "https://huggingface.co/$1/resolve/$2/$3" -o "weights/$4"
}
hf_file robbyant/lingbot-map-v2-base-streaming e6a22da8d071200b25dd3bfde2bd499a230819ee model_weights.pt LingBot2.pt
hf_file robbyant/lingbot-map-v2-base-streaming e6a22da8d071200b25dd3bfde2bd499a230819ee config.json LingBot2.json
hf_file robbyant/lingbot-map e8984048a940fbdfa19909f8cc0c0ad834c1ee5a lingbot-map.pt LingBot1.pt
hf_file facebook/VGGT-1B 860abec7937da0a4c03c41d3c269c366e82abdf9 model.safetensors VGGT.safetensors
hf_file yyfz233/Pi3 ae722e7039287d0c8fde9f11f197f804f44b510c model.safetensors Pi3.safetensors
```

Pass the pinned source checkouts directly rather than installing upstream demo extras or CUDA wheels. The download names above are local; keep the 2.0 JSON and checkpoint together. Run models in separate processes, sequentially, to avoid timing and memory contention:

```sh
run() {
  python run_model.py "$1" --source "data/$2" --weights "weights/$3" \
    --dataset data/rgbd_dataset_freiburg1_desk --output "outputs/$1.npz"
}
run lingbot2 lingbot-map-v2 LingBot2.pt
run vggt vggt VGGT.safetensors
run pi3 Pi3 Pi3.safetensors
run lingbot1 lingbot-map LingBot1.pt
python evaluate.py outputs/lingbot2.npz outputs/vggt.npz outputs/pi3.npz outputs/lingbot1.npz \
  --dataset data/rgbd_dataset_freiburg1_desk --output outputs --figure ~/bench_compare.png
```

The evaluator writes `metrics.json`, RGB binary PLYs for the reference and each model, and a single oblique comparison with identical plot bounds and view. These are local outputs, not tracked fixtures.

## Method and limits

- The shared clip is 24 evenly time-spaced frames over the first 19.836 s with RGB/depth associations. Depth is nearest-time within 20 ms (actual max 17.42 ms); reference poses are timestamp-interpolated with linear translation and quaternion SLERP. The whole RGB field of view is resized to 512×384 for 2.0 (16px patches) and 518×392 for the others (14px patches). Both streams initialize on the first eight frames and retain all frames; VGGT and π3 see the entire sequence at once.
- Pose scores are translation ATE RMSE and consecutive-pair translation and rotation RPE RMSE after one orientation-preserving Sim(3) fit to camera centers. Depth is averaged over frames after separate per-frame median scale alignment of jointly valid depths. Cloud scores **do not** use that per-frame alignment: we back-project model depth with predicted intrinsics (or use π3's native camera-frame local points), transform by predicted camera-to-world poses, and apply only the global trajectory Sim(3). The independent GT cloud uses TUM RGB calibration and registered depth. Sample roughly every fifth native image pixel (every fourth resized pixel), then retain one point per 1 cm voxel. Accuracy and completeness are directed mean nearest-neighbor distances; Chamfer-L1 is their mean. No model-specific confidence thresholds or outlier rejection are used for the scores; the figure clips to shared reference bounds.
- Wall time includes prediction/output normalization; it excludes image preparation, checkpoint loading, evaluation and export. Peak RSS is the Linux process high-water mark including model loading. These are single-run CPU numbers, not mobile or vendor GPU throughput. FP32 is used on CPU. π3 uses its PyTorch RoPE fallback and PyTorch's fused CPU SDPA for its decoder to avoid materializing quadratic attention matrices.
- The official LingBot-Map 2.0 loader requires CUDA BF16 and the online runtime uses FlashInfer. Its released causal weights were evaluated here with the native model, heads and streaming controller but a short-clip FP32 PyTorch SDPA KV replacement. No cache eviction was needed. CUDA parity is not established. The [project page](https://technology.robbyant.com/lingbot-map-v2) describes context reconsolidation, but the published weights and API expose only the causal streaming release, so reconsolidation was not benchmarked.
- A desk clip is not a scanner acceptance test. Deployment would need phone captures, more scenes and a longer revisit clip, surface/mesh checks, and license review. The 2.0 research checkpoint and original VGGT-1B and π3 checkpoints used here are non-commercial; Meta offers a separate VGGT commercial checkpoint, not tested here.
