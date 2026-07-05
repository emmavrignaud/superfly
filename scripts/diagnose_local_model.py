#!/usr/bin/env python3
"""Step-by-step local RF-DETR diagnostic — run in a terminal, not Jupyter.

Jupyter kernel death = native crash (segfault / OpenMP), not a Python exception.
This script prints each step and flushes so you see exactly where it dies.

Usage (from repo root):
    conda activate fly-tracking
    python scripts/diagnose_local_model.py
"""
from __future__ import annotations

import os
import sys

# Windows: conda-forge (llvm libomp) + PyTorch/MKL (libiomp5) in one env triggers
# OMP Error #15 and hard-exits before Python can catch anything. Set before torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = REPO_ROOT / "RF-DETR_model" / "weights.pt"


def step(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def main() -> int:
    step("1. Python")
    print(sys.executable, flush=True)
    print(sys.version, flush=True)

    step("2. torch")
    import torch
    from transformers.utils import is_torch_available

    print(f"torch {torch.__version__}", flush=True)
    print(f"transformers sees torch: {is_torch_available()}", flush=True)
    if not is_torch_available():
        print("FAIL: need torch >= 2.4 for transformers 5.x", flush=True)
        return 1

    step("3. transformers BackboneConfigMixin")
    from transformers import BackboneConfigMixin  # noqa: F401

    print("BackboneConfigMixin OK", flush=True)

    step("4. rfdetr_plus import")
    from rfdetr_plus import RFDETR2XLarge

    print("RFDETR2XLarge OK", flush=True)

    step("5. weights file")
    if not WEIGHTS.exists():
        print(f"FAIL: missing {WEIGHTS}", flush=True)
        return 1
    print(f"{WEIGHTS} ({WEIGHTS.stat().st_size / 1e6:.0f} MB)", flush=True)

    step("6. load model (this is where kernels usually die — wait)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    # Roboflow exports often lack model_name in args — from_checkpoint cannot infer
    # the variant (pretrain_weights=/train/cache/... is useless). Use the class directly.
    model = RFDETR2XLarge(
        pretrain_weights=str(WEIGHTS),
        accept_platform_model_license=True,
        resolution=640,
    )
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    print(f"class_names: {getattr(model, 'class_names', '?')}", flush=True)

    step("7. one predict on zeros")
    import numpy as np

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with torch.inference_mode():
        dets = model.predict(frame, threshold=0.1)
    print(f"detections: {len(dets)}", flush=True)

    step("DONE — local stack works; if notebook still dies, Jupyter/OpenMP is the culprit")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nPython exception: {exc}", flush=True)
        raise
