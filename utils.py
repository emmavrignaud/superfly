"""
utils.py

Shared helpers for scripts and notebooks.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path


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
