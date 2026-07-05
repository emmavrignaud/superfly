"""
utils.py

Shared helpers for scripts and notebooks.
"""

import json
import os
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


def make_run_output_dir(
    raw_video: str | Path,
    outputs_root: str | Path = "data/outputs",
) -> str:
    """Auto-incremented run directory: data/outputs/run_N_<DPE>DPE_n<NNN>.

    Suffix is derived from the "<N> DPE/<NNN>" pattern in the video path
    (e.g. "13 DPE/002" -> "13DPE_n002"). Falls back to the video stem
    (truncated to 20 chars) when the pattern is absent. The numeric N is one
    more than the largest existing run_* directory under outputs_root.
    """
    import re
    raw_video = str(raw_video)
    outputs_root = Path(outputs_root)
    outputs_root.mkdir(parents=True, exist_ok=True)

    m = re.search(r"(\d+)\s+DPE[/\\](\d+)", raw_video)
    suffix = f"{m.group(1)}DPE_n{m.group(2).zfill(3)}" if m else Path(raw_video).stem[:20]

    existing = [d for d in outputs_root.iterdir() if d.is_dir() and d.name.startswith("run_")]
    next_n = max(
        (int(d.name.split("_")[1]) for d in existing if d.name.split("_")[1].isdigit()),
        default=0,
    ) + 1
    out = outputs_root / f"run_{next_n}_{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def save_config_snapshot(
    out_dir: str | Path,
    config_path: str | Path = "config.yaml",
) -> None:
    """Copy the active config.yaml verbatim into the run directory.

    Pairs with save_run_params: per-stage outputs go to run_params.json,
    the full config snapshot goes to config.yaml beside it.
    """
    src = Path(config_path)
    dst = Path(out_dir) / "config.yaml"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


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


def load_creds(config, creds_path: str | Path = "creds_config.yaml") -> tuple[str, str]:
    """Return (api_key, model_id) from creds_config.yaml.

    Secrets live only in creds_config.yaml (git-ignored). MODEL_ID falls back to
    config.roboflow.model_id when the creds file doesn't set it. Raises
    SystemExit with a clear message when the file or API_KEY is missing, so both
    scripts fail the same way instead of each rolling their own loader.
    """
    creds_path = Path(creds_path)
    if not creds_path.exists():
        raise SystemExit(f"creds_config.yaml not found at {creds_path}")
    with open(creds_path) as f:
        creds = yaml.safe_load(f) or {}
    api_key  = creds.get("API_KEY", "")
    model_id = creds.get("MODEL_ID") or config.roboflow.model_id
    if not api_key:
        raise SystemExit("API_KEY missing in creds_config.yaml")
    return api_key, model_id


def link_or_copy(src: str | Path, dst: str | Path) -> None:
    """Hardlink src -> dst to save disk; fall back to a real copy if the
    filesystem refuses (e.g. across drives). No-op if dst already exists."""
    import shutil
    src, dst = Path(src), Path(dst)
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
