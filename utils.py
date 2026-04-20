"""
utils.py

Shared helpers for scripts and notebooks.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path


def resolve_overlay_video(
    run_dir: str | Path,
    overlay_source: str,
    override: str | None = None,
) -> str | None:
    """Pick the video path to pass into render_*_overlay_video.

    Decision order:
      1. If ``override`` is given, return it as-is (caller's explicit --video).
      2. If ``overlay_source == "processed"``, return run_params.preprocessing.video_pp.
      3. Otherwise (``raw_cropped`` or unknown), return run_params.config.video (raw).

    The resolved path is searched under the run directory (where _pp lives and
    where the raw is hardlinked) and under repo_root/notebooks/ (where config.video
    is stored relative to the notebook cwd). Returns None if nothing exists on disk.
    """
    if override is not None:
        return override

    run_dir = Path(run_dir)
    params_path = run_dir / "run_params.json"
    if not params_path.exists():
        return None

    with open(params_path) as f:
        params = json.load(f)

    mode = (overlay_source or "raw_cropped").lower()
    if mode == "processed":
        vid = params.get("preprocessing", {}).get("video_pp")
    else:
        vid = params.get("config", {}).get("video")

    if not vid:
        return None

    repo_root = Path(__file__).resolve().parent
    vid_p = Path(vid)
    candidates = [
        vid_p if vid_p.is_absolute() else None,
        run_dir / vid_p.name,
        (repo_root / "notebooks" / vid_p).resolve(),
    ]
    for c in candidates:
        if c is not None and c.exists():
            return str(c)
    return None


def save_run_params(out_dir: str, stage: str, params: dict) -> None:
    """
    Incrementally merge `params` under `stage` into outputs/run_N/run_params.json.

    Top-level metadata (timestamp, git_commit) is written on the first call only.
    Subsequent calls from later pipeline stages add their own key, so a partial
    run still captures everything that completed.

    Parameters
    ----------
    out_dir : str | Path  — the run output directory (e.g. outputs/run_5)
    stage   : str         — key under which params are stored (e.g. "tracker")
    params  : dict        — JSON-serialisable dict of parameters / output paths
    """
    path = Path(out_dir) / "run_params.json"

    data: dict = {}
    if path.exists():
        with open(path) as f:
            data = json.load(f)

    if "timestamp" not in data:
        data["timestamp"] = datetime.now().isoformat(timespec="seconds")

    if "git_commit" not in data:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True,
                cwd=Path(__file__).parent,
            )
            data["git_commit"] = result.stdout.strip() or "unknown"
        except Exception:
            data["git_commit"] = "unknown"

    data[stage] = params

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
