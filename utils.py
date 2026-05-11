"""
utils.py

Shared helpers for scripts and notebooks.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


class Config(dict):
    """A dict that also supports attribute-style access (cfg.section.key).

    Constructed eagerly: any nested dict in the input becomes a Config too,
    and any list of dicts becomes a list of Configs. Missing keys raise
    AttributeError (the contract is enforced by tests/test_config_schema.py;
    fallbacks live in config.yaml itself, not at every call site).

    Still a real dict, so existing code that does ``cfg["section"]`` or
    ``cfg.get("section")`` keeps working.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            if isinstance(v, dict) and not isinstance(v, Config):
                self[k] = Config(v)
            elif isinstance(v, list):
                self[k] = [Config(x) if isinstance(x, dict) else x for x in v]

    def __getattr__(self, key: str):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(
                f"Config has no key {key!r}. "
                f"Available: {sorted(self.keys())}"
            ) from e


def load_config(path: str | Path) -> Config:
    """Load a YAML file and return it wrapped in a Config for attribute access."""
    with open(path) as f:
        return Config(yaml.safe_load(f) or {})


def resolve_overlay_video(
    run_dir: str | Path,
    overlay_source: str,
    override: str | None = None,
) -> str | None:
    """Pick the video path to pass into render_*_overlay_video.

    Decision order:
      1. If ``override`` is given, return it as-is (caller's explicit --video).
      2. If ``overlay_source == "processed"``, return run_params.preprocessing.video_pp.
      3. If ``overlay_source == "raw_cropped"``, prefer
         run_params.preprocessing.video_raw_cropped when present.
      4. Otherwise (``raw_cropped`` fallback or unknown), return
         run_params.config.video (raw).

    The resolved path is searched under the run directory (where _pp, raw_cropped,
    and the hardlinked raw live) and under repo_root/notebooks/ (where config.video
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
    elif mode == "raw_cropped":
        vid = (
            params.get("preprocessing", {}).get("video_raw_cropped")
            or params.get("config", {}).get("video")
        )
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
