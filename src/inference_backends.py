"""Fly detection backends used by collect_detections().

What this module does
---------------------
Runs RF-DETR on each video frame and returns bounding boxes. Two backends:

  hosted — sends each frame to Roboflow's cloud API (default; no GPU needed).
  local  — loads weights.pt on this machine (needs GPU for reasonable speed).

Why it exists
-------------
The tracking pipeline should not care *where* boxes come from. This module
hides hosted vs local so tracking.py stays one code path.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import supervision as sv

# Windows: avoid hard crash when conda OpenMP and PyTorch MKL both load.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


class DetectionBackend(ABC):
    """Run RF-DETR on one BGR frame."""

    @abstractmethod
    def predict(self, bgr: np.ndarray) -> sv.Detections:
        ...


class HostedBackend(DetectionBackend):
    """Roboflow hosted API (InferenceHTTPClient)."""

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        model_id: str,
        threshold: float,
    ) -> None:
        from utils import patch_windows_ssl

        patch_windows_ssl()
        from inference_sdk import InferenceHTTPClient
        from inference_sdk.http.entities import InferenceConfiguration

        self._model_id = model_id
        client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
        client.configure(InferenceConfiguration(confidence_threshold=threshold))
        self._client = client

    def predict(self, bgr: np.ndarray) -> sv.Detections:
        result = self._client.infer(bgr, model_id=self._model_id)
        return sv.Detections.from_inference(result)


class LocalBackend(DetectionBackend):
    """Local RF-DETR from a weights file on disk."""

    def __init__(
        self,
        *,
        weights_path: Path,
        resolution: int,
        num_classes: int,
        threshold: float,
        optimize_for_gpu: bool,
    ) -> None:
        import torch
        from rfdetr_plus import RFDETR2XLarge

        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Local weights not found: {weights_path}\n"
                "Download or copy weights.pt into place, or set inference.local.weights_path."
            )

        self._threshold = threshold
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"Loading local RF-DETR from {weights_path} on {device} …", flush=True)

        model = RFDETR2XLarge(
            pretrain_weights=str(weights_path),
            accept_platform_model_license=True,
            resolution=resolution,
            num_classes=num_classes,
        )
        if hasattr(model, "to"):
            model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        if optimize_for_gpu and device == "cuda" and hasattr(model, "optimize_for_inference"):
            model.optimize_for_inference(dtype=torch.float16)
            print("  optimize_for_inference(FP16) enabled", flush=True)
        elif optimize_for_gpu and device not in ("cuda",):
            print(f"  optimize_for_gpu ignored ({device} device)", flush=True)

        self._model = model
        self._device = device

    def predict(self, bgr: np.ndarray) -> sv.Detections:
        import torch

        with torch.inference_mode():
            return self._model.predict(bgr, threshold=self._threshold)


def create_detection_backend(
    mode: str,
    *,
    repo_root: Path,
    api_url: str,
    api_key: str,
    model_id: str,
    threshold: float,
    local_weights_path: str,
    local_resolution: int,
    local_num_classes: int,
    local_optimize_for_gpu: bool,
) -> DetectionBackend:
    """Build the backend named by inference.mode in config.yaml."""
    mode = (mode or "hosted").strip().lower()
    if mode == "hosted":
        if not api_key:
            raise ValueError(
                "inference.mode is 'hosted' but API_KEY is missing in creds_config.yaml"
            )
        return HostedBackend(
            api_url=api_url,
            api_key=api_key,
            model_id=model_id,
            threshold=threshold,
        )
    if mode == "local":
        weights = Path(local_weights_path)
        if not weights.is_absolute():
            weights = repo_root / weights
        return LocalBackend(
            weights_path=weights,
            resolution=local_resolution,
            num_classes=local_num_classes,
            threshold=threshold,
            optimize_for_gpu=local_optimize_for_gpu,
        )
    raise ValueError(
        f"inference.mode must be 'hosted' or 'local', got {mode!r}"
    )
