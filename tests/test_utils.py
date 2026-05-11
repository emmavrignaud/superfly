"""
Unit tests for utils.save_run_params.

Documents and pins the merging behavior the pipeline depends on: each stage
adds its own key without clobbering previous stages, and metadata
(``timestamp``, ``git_commit``) is written only on the first call.
"""

from __future__ import annotations

import json
from pathlib import Path

from utils import save_run_params


def test_first_call_creates_file_with_metadata(tmp_path: Path) -> None:
    save_run_params(str(tmp_path), "stage_one", {"foo": 1})
    path = tmp_path / "run_params.json"
    assert path.exists()

    data = json.loads(path.read_text())
    assert data["stage_one"] == {"foo": 1}
    assert "timestamp" in data
    assert "git_commit" in data


def test_subsequent_calls_merge_under_new_keys(tmp_path: Path) -> None:
    save_run_params(str(tmp_path), "stage_one", {"foo": 1})
    save_run_params(str(tmp_path), "stage_two", {"bar": [1, 2, 3]})

    data = json.loads((tmp_path / "run_params.json").read_text())
    assert data["stage_one"] == {"foo": 1}
    assert data["stage_two"] == {"bar": [1, 2, 3]}


def test_metadata_is_not_overwritten(tmp_path: Path) -> None:
    save_run_params(str(tmp_path), "stage_one", {"foo": 1})
    first = json.loads((tmp_path / "run_params.json").read_text())

    save_run_params(str(tmp_path), "stage_two", {"bar": 2})
    second = json.loads((tmp_path / "run_params.json").read_text())

    assert first["timestamp"] == second["timestamp"], "timestamp must be set once"
    assert first["git_commit"] == second["git_commit"], "git_commit must be set once"


def test_stage_overwrite_replaces_old_value(tmp_path: Path) -> None:
    """Calling the same stage twice replaces the previous params for that stage
    (this is the documented contract: stages are keys, not append lists)."""
    save_run_params(str(tmp_path), "tracker", {"version": "v1"})
    save_run_params(str(tmp_path), "tracker", {"version": "v2"})
    data = json.loads((tmp_path / "run_params.json").read_text())
    assert data["tracker"] == {"version": "v2"}
