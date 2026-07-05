#!/usr/bin/env python3
"""Compare local RF-DETR vs Roboflow hosted on the same frames.

Usage (repo root, fly-tracking env):
    python scripts/test_local_model.py
    python scripts/test_local_model.py --video path/to/clip.mp4 --benchmark 30
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WEIGHTS_PATH = REPO_ROOT / "RF-DETR_model" / "weights.pt"
CONFIG_PATH = REPO_ROOT / "config.yaml"
CREDS_PATH = REPO_ROOT / "creds_config.yaml"
CHECKPOINT_RESOLUTION = 640
FRAME_FRACTIONS = (0.0, 0.25, 0.5, 0.9)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local vs hosted RF-DETR timing compare")
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--benchmark", type=int, default=30, metavar="N",
                   help="Frames from start to time (default 30, 0=skip)")
    p.add_argument("--local-only", action="store_true")
    p.add_argument("--hosted-only", action="store_true")
    return p.parse_args()


def detections_to_xyxyc(dets):
    import numpy as np
    if dets is None or len(dets) == 0 or dets.confidence is None:
        return np.empty((0, 5), dtype=np.float32)
    return np.hstack([dets.xyxy, dets.confidence[:, None]]).astype(np.float32)


def read_spot_frames(video_path: Path, fractions: tuple[float, ...]):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    indices = sorted({min(int(round(f * (n_total - 1))), n_total - 1) for f in fractions})
    frames = {}
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if ok:
            frames[idx] = bgr
    cap.release()
    return frames, img_w, img_h, n_total, fps


def read_consecutive_frames(video_path: Path, n: int):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    out = []
    while len(out) < n:
        ok, bgr = cap.read()
        if not ok:
            break
        out.append(bgr)
    cap.release()
    return out


def load_local_model(device: str):
    from rfdetr_plus import RFDETR2XLarge

    if not WEIGHTS_PATH.exists():
        raise SystemExit(f"Missing weights: {WEIGHTS_PATH}")
    print(f"Loading local model …", flush=True)
    t0 = time.perf_counter()
    model = RFDETR2XLarge(
        pretrain_weights=str(WEIGHTS_PATH),
        accept_platform_model_license=True,
        resolution=CHECKPOINT_RESOLUTION,
        num_classes=1,
    )
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s on {device}")
    return model


def load_hosted_client(config, threshold: float):
    from utils import load_config, patch_windows_ssl

    patch_windows_ssl()
    from inference_sdk import InferenceHTTPClient
    from inference_sdk.http.entities import InferenceConfiguration

    if not CREDS_PATH.exists():
        raise SystemExit("Hosted compare needs creds_config.yaml")
    creds = load_config(CREDS_PATH)
    api_key = getattr(creds, "API_KEY", "") or ""
    if not api_key:
        raise SystemExit("API_KEY missing in creds_config.yaml")
    model_id = getattr(creds, "MODEL_ID", "") or config.roboflow.model_id

    print(f"Connecting hosted API ({config.roboflow.inference_api_url}, {model_id}) …", flush=True)
    client = InferenceHTTPClient(
        api_url=config.roboflow.inference_api_url,
        api_key=api_key,
    )
    client.configure(InferenceConfiguration(confidence_threshold=threshold))
    return client, model_id


def infer_local(model, bgr, threshold: float):
    import torch

    with torch.inference_mode():
        t0 = time.perf_counter()
        dets = model.predict(bgr, threshold=threshold)
        ms = (time.perf_counter() - t0) * 1000
    return dets, ms


def infer_hosted(client, bgr, model_id: str):
    import supervision as sv

    t0 = time.perf_counter()
    result = client.infer(bgr, model_id=model_id)
    ms = (time.perf_counter() - t0) * 1000
    dets = sv.Detections.from_inference(result)
    return dets, ms


def run_spot_compare(frames, *, do_local, do_hosted, model, client, model_id, threshold):
    print(f"\n{'frame':>6}  {'local_n':>7}  {'local_ms':>9}  {'hosted_n':>8}  {'hosted_ms':>10}")
    print("-" * 50)
    local_total_ms = hosted_total_ms = 0.0
    for idx, bgr in sorted(frames.items()):
        local_n = local_ms = hosted_n = hosted_ms = None
        if do_local:
            dets, local_ms = infer_local(model, bgr, threshold)
            local_n = len(detections_to_xyxyc(dets))
            local_total_ms += local_ms
        if do_hosted:
            dets, hosted_ms = infer_hosted(client, bgr, model_id)
            hosted_n = len(detections_to_xyxyc(dets))
            hosted_total_ms += hosted_ms
        ln = f"{local_n:>7}" if local_n is not None else "      -"
        lm = f"{local_ms:>8.0f}" if local_ms is not None else "       -"
        hn = f"{hosted_n:>8}" if hosted_n is not None else "       -"
        hm = f"{hosted_ms:>9.0f}" if hosted_ms is not None else "        -"
        print(f"{idx:>6}  {ln}  {lm}  {hn}  {hm}")
    n = len(frames)
    print("-" * 50)
    if do_local:
        print(f"  local  spot total: {local_total_ms:.0f} ms  ({local_total_ms / n:.0f} ms/frame)")
    if do_hosted:
        print(f"  hosted spot total: {hosted_total_ms:.0f} ms  ({hosted_total_ms / n:.0f} ms/frame)")


def run_benchmark(frames_list, n_total, *, do_local, do_hosted, model, client, model_id, threshold):
    import numpy as np

    n = len(frames_list)
    if n == 0:
        return
    print(f"\n--- benchmark ({n} consecutive frames) ---")

    local_elapsed = local_counts = None
    if do_local:
        import torch

        with torch.inference_mode():
            t0 = time.perf_counter()
            local_counts = [len(model.predict(bgr, threshold=threshold)) for bgr in frames_list]
            local_elapsed = time.perf_counter() - t0
        local_ms = local_elapsed / n * 1000
        print(f"  LOCAL:  {local_elapsed:.1f}s total | {local_ms:.0f} ms/frame | "
              f"{1000 / local_ms:.2f} fps | dets mean={np.mean(local_counts):.1f}")
        print(f"          full video ({n_total} frames): ~{local_ms * n_total / 1000 / 60:.1f} min")

    hosted_elapsed = hosted_counts = None
    if do_hosted:
        import supervision as sv

        t0 = time.perf_counter()
        hosted_counts = []
        for bgr in frames_list:
            result = client.infer(bgr, model_id=model_id)
            hosted_counts.append(len(sv.Detections.from_inference(result)))
        hosted_elapsed = time.perf_counter() - t0
        hosted_ms = hosted_elapsed / n * 1000
        print(f"  HOSTED: {hosted_elapsed:.1f}s total | {hosted_ms:.0f} ms/frame | "
              f"{1000 / hosted_ms:.2f} fps | dets mean={np.mean(hosted_counts):.1f}")
        print(f"          full video ({n_total} frames): ~{hosted_ms * n_total / 1000 / 60:.1f} min")

    if do_local and do_hosted:
        ratio = local_elapsed / hosted_elapsed
        faster = "hosted" if ratio > 1 else "local"
        print(f"  => {faster} is {max(ratio, 1 / ratio):.1f}x faster on this machine")


def main() -> int:
    args = parse_args()
    if args.local_only and args.hosted_only:
        raise SystemExit("Pick one of --local-only or --hosted-only, not both")

    import torch
    from utils import load_config

    config = load_config(CONFIG_PATH)
    threshold = float(config.tracker.detection_confidence_rfdetr)
    video_path = args.video or (REPO_ROOT / config.video.raw_path)
    if not video_path.is_absolute():
        video_path = REPO_ROOT / video_path
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    do_local = not args.hosted_only
    do_hosted = not args.local_only
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"torch {torch.__version__} | device {device}")
    print(f"video {video_path} | threshold {threshold}")
    print(f"modes: local={'yes' if do_local else 'no'} hosted={'yes' if do_hosted else 'no'}")

    model = client = model_id = None
    if do_local:
        model = load_local_model(device)
    if do_hosted:
        client, model_id = load_hosted_client(config, threshold)

    spot_frames, img_w, img_h, n_total, fps = read_spot_frames(video_path, FRAME_FRACTIONS)
    print(f"{img_w}x{img_h}, {n_total} frames @ {fps:.1f} fps | spot: {sorted(spot_frames.keys())}")

    run_spot_compare(
        spot_frames,
        do_local=do_local,
        do_hosted=do_hosted,
        model=model,
        client=client,
        model_id=model_id,
        threshold=threshold,
    )

    if args.benchmark > 0:
        bench_frames = read_consecutive_frames(video_path, min(args.benchmark, n_total))
        run_benchmark(
            bench_frames,
            n_total,
            do_local=do_local,
            do_hosted=do_hosted,
            model=model,
            client=client,
            model_id=model_id,
            threshold=threshold,
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
