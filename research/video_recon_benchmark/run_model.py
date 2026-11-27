import argparse
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from tum import load_sequence


MODELS = {
    "lingbot2": (512, 384),
    "lingbot1": (518, 392),
    "vggt": (518, 392),
    "pi3": (518, 392),
}


class ShortClipCPUCache:
    """FP32 SDPA stand-in for FlashInfer; only no-eviction clips are supported."""

    dtype = torch.float32

    def __init__(self, *, layers, capacity, scale, window, **_):
        if capacity > scale + window:
            raise ValueError("CPU cache does not implement eviction")
        self.committed = [[] for _ in range(layers)]
        self.pending = None

    def begin(self):
        self.pending = {}

    def write(self, layer, key, value):
        self.pending[layer] = (key.contiguous(), value.contiguous())

    def run(self, layer, query):
        key, value = (torch.cat(items, dim=0) for items in
                      zip(*self.committed[layer], self.pending[layer]))
        q, k, v = (x.transpose(0, 1)[None].contiguous() for x in (query, key, value))
        return F.scaled_dot_product_attention(q, k, v)[0].transpose(0, 1)

    def commit(self):
        for layer, item in self.pending.items():
            self.committed[layer].append(item)
        self.rollback()

    def rollback(self):
        self.pending = None


def load_lingbot2(weights, width, height, count, device):
    from lingbot_map2 import load_model
    from lingbot_map2.aggregator import stream as api
    from lingbot_map2.models.gcr_base import NativeModel
    from lingbot_map2.models.gcr_stream import GCRStream
    from lingbot_map2.utils.checkpoint import resolve_checkpoint

    if device == "cpu":
        path, spec, _ = resolve_checkpoint(weights, weights.with_suffix(".json"))
        native = NativeModel(spec.spatial_rope, max_frames=count)
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if state.get("training_mode", "causal") != "causal":
            raise ValueError("expected a causal checkpoint")
        native.load_state_dict(state.get("model", state), strict=True)
        model = GCRStream(native.eval(), replace(spec, max_frames=count), width=width, height=height)
    else:
        model = load_model(weights, config=weights.with_suffix(".json"), width=width, height=height,
                           device=device, compile=False, max_frames=count)
    if count < model.config.scale_frames:
        raise ValueError("not enough frames for the LingBot-Map 2.0 initialization prefix")

    def predict(images):
        chunks = []
        cache = ShortClipCPUCache if device == "cpu" else api.PagedKVCache
        with patch.object(api, "PagedKVCache", cache), model.create_stream() as stream:
            for image in images:
                out = stream.online(image)
                if out is not None:
                    chunks.append(out)
        if not np.array_equal(np.concatenate([c["frame_indices"] for c in chunks]), np.arange(count)):
            raise ValueError("LingBot-Map 2.0 output is missing frames")
        return {key: np.concatenate([c[key] for c in chunks]) for key in
                ("poses_c2w", "intrinsics", "depth")}

    return predict


def numpy_predictions(pose, depth, **fields):
    return {key: value.detach().float().cpu().numpy() for key, value in
            dict(poses_c2w=pose, depth=depth, **fields).items()}


def homogeneous(pose):
    result = torch.eye(4, dtype=pose.dtype, device=pose.device).repeat(len(pose), 1, 1)
    result[:, :3] = pose
    return result


def load_lingbot1(weights, width, device):
    from lingbot_map.models.gct_stream import GCTStream
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

    model = GCTStream(img_size=width, patch_size=14, enable_3d_rope=True, max_frame_num=1024,
                      kv_cache_sliding_window=64, kv_cache_scale_frames=8,
                      kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
                      use_sdpa=True, camera_num_iterations=4)
    state = torch.load(weights, map_location="cpu", mmap=True, weights_only=True)
    state = state.get("model", state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or any(not key.startswith("point_head.") for key in unexpected):
        raise ValueError(f"unexpected checkpoint keys: missing={missing}, unexpected={unexpected[:10]}")
    model.to(device).eval()

    def predict(images):
        out = model.inference_streaming(images, num_scale_frames=8, keyframe_interval=1)
        pose, intrinsics = pose_encoding_to_extri_intri(out["pose_enc"], images.shape[-2:])
        return numpy_predictions(homogeneous(pose[0]), out["depth"][0], intrinsics=intrinsics[0])

    return predict


def load_vggt(weights, device):
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    model = VGGT(enable_point=False, enable_track=False)
    state = load_file(weights)
    state = {key: val for key, val in state.items() if not key.startswith(("point_head.", "track_head."))}
    model.load_state_dict(state)
    model.to(device).eval()

    def predict(images):
        out = model(images)
        pose, intrinsics = pose_encoding_to_extri_intri(out["pose_enc"], images.shape[-2:])
        return numpy_predictions(torch.linalg.inv(homogeneous(pose[0])), out["depth"][0],
                                 intrinsics=intrinsics[0])

    return predict


def load_pi3(weights, device):
    from pi3.models.layers import attention
    from pi3.models.pi3 import Pi3
    from torch.nn.attention import SDPBackend, sdpa_kernel

    model = Pi3()
    model.load_state_dict(load_file(weights))
    model.to(device).eval()

    def cpu_attention(q, k, v, **kwargs):
        # Upstream selects quadratic-memory math attention for FP32.
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(), **kwargs)

    def predict(images):
        if device == "cpu":
            with patch.object(attention, "scaled_dot_product_attention", cpu_attention):
                out = model(images[None])
        else:
            out = model(images[None])
        return numpy_predictions(out["camera_poses"][0], out["local_points"][0, ..., 2:3],
                                 local_points=out["local_points"][0])

    return predict


def main():
    parser = argparse.ArgumentParser(description="Run one model on the shared TUM RGB-D desk clip")
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--source", type=Path, required=True, help="upstream model source checkout")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not args.source.is_dir() or not args.weights.is_file():
        parser.error("--source and --weights must point to an existing checkout and checkpoint")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    sys.path.insert(0, str(args.source.resolve()))
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    width, height = MODELS[args.model]
    sequence = load_sequence(args.dataset, args.frames, args.seconds, width, height)
    images = torch.from_numpy(sequence["rgb"]).permute(0, 3, 1, 2).contiguous().float() / 255
    if args.model != "lingbot2":
        images = images.to(args.device)
    print(f"Loading {args.model}; {len(images)} frames at {width}×{height}, {args.device}", flush=True)
    if args.model == "lingbot2":
        predict = load_lingbot2(args.weights, width, height, args.frames, args.device)
    elif args.model == "lingbot1":
        predict = load_lingbot1(args.weights, width, args.device)
    elif args.model == "vggt":
        predict = load_vggt(args.weights, args.device)
    else:
        predict = load_pi3(args.weights, args.device)
    print("Running inference", flush=True)
    start = time.perf_counter()
    autocast = args.device == "cuda" and args.model != "lingbot2"
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        prediction = predict(images)
    if args.device == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    print(f"{args.model}: {wall:.2f} s, peak RSS {peak:.3f} GiB", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **sequence, **prediction, wall_seconds=wall, peak_rss_gib=peak)


if __name__ == "__main__":
    main()
