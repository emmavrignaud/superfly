"""Check local defaults and explicit hosted opt-in without loading a model or contacting an API."""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

import pytest

from utils import Config, inference_requires_api_key, inference_tracking_kwargs, load_creds


def _load_module(repo_root, relative_path):
    spec = importlib.util.spec_from_file_location("inference_default_test_module", repo_root / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", [None, "", "local"])
def test_local_defaults_need_no_credentials(mode, tmp_path):
    inference = {"local": {"weights_path": "weights.pt", "resolution": 640,
                            "num_classes": 1, "optimize_for_gpu": False}}
    if mode is not None:
        inference["mode"] = mode
    config = Config({"inference": inference, "roboflow": {"model_id": "optional-model/1"}})

    assert not inference_requires_api_key(config)
    assert inference_tracking_kwargs(config, tmp_path)["inference_mode"] == "local"
    assert load_creds(config, tmp_path / "missing.yaml", require_api_key=False) == ("", "optional-model/1")


@pytest.fixture
def backends(repo_root, monkeypatch):
    # Only the factory is under test; the detection library and model constructors are not needed.
    monkeypatch.setitem(sys.modules, "supervision", ModuleType("supervision"))
    return _load_module(repo_root, "src/inference_backends.py")


@pytest.fixture
def backend_kwargs(repo_root):
    return dict(repo_root=repo_root, api_url="https://example.invalid", api_key="", model_id="model/1",
                threshold=0.1, local_weights_path="weights.pt", local_resolution=640,
                local_num_classes=1, local_optimize_for_gpu=False)


@pytest.mark.parametrize("mode_kwargs", [{}, {"mode": None}, {"mode": ""}])
def test_backend_factory_defaults_to_local(backends, backend_kwargs, mode_kwargs, monkeypatch):
    sentinel = object()
    monkeypatch.setattr(backends, "LocalBackend", lambda **kwargs: sentinel)
    monkeypatch.setattr(backends, "HostedBackend", lambda **kwargs: pytest.fail("Hosted inference was not requested"))

    assert backends.create_detection_backend(**backend_kwargs, **mode_kwargs) is sentinel


def test_explicit_hosted_mode_remains_available(backends, backend_kwargs, monkeypatch):
    sentinel = object()
    monkeypatch.setattr(backends, "HostedBackend", lambda **kwargs: sentinel)
    backend_kwargs["api_key"] = "test-key"

    assert backends.create_detection_backend("hosted", **backend_kwargs) is sentinel
    assert inference_requires_api_key(Config({"inference": {"mode": "hosted"}}))


def test_hosted_mode_requires_credentials(backends, backend_kwargs):
    with pytest.raises(ValueError, match="API_KEY"):
        backends.create_detection_backend("hosted", **backend_kwargs)
