"""Tests for the whisper.cpp backend that need no model and no GPU."""

from pathlib import Path
from unittest.mock import patch

import pytest

from wyoming_vulkan_whisper.device import (
    device_index,
    is_gpu,
    is_vulkan,
    whisper_cpp_gpu,
)

# --- --device ---------------------------------------------------------------


@pytest.mark.parametrize("device", ["vulkan", "vulkan:1", "VULKAN", " vulkan "])
def test_vulkan_is_recognized(device) -> None:
    assert is_vulkan(device)
    assert is_gpu(device)


@pytest.mark.parametrize("device", ["cpu", "CPU", "auto", ""])
def test_everything_else_runs_on_the_cpu(device) -> None:
    assert not is_vulkan(device)
    assert not is_gpu(device)


def test_index_is_parsed() -> None:
    assert device_index("vulkan") is None
    assert device_index("vulkan:2") == 2
    assert device_index("vulkan:first") is None


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("cpu", (False, 0)),
        ("auto", (False, 0)),
        ("vulkan", (True, 0)),
        ("vulkan:1", (True, 1)),
        # whisper.cpp uses whatever GPU backend it was built with.
        ("cuda", (True, 0)),
        ("cuda:2", (True, 2)),
    ],
)
def test_whisper_cpp_gpu(device, expected) -> None:
    assert whisper_cpp_gpu(device) == expected


# --- model resolution (needs pywhispercpp, but no download) --------------------


def test_model_path_is_used_as_is(tmp_path: Path) -> None:
    from wyoming_vulkan_whisper.whisper_cpp_handler import resolve_ggml_model

    model = tmp_path / "ggml-custom.bin"
    model.write_bytes(b"")
    assert resolve_ggml_model(str(model), tmp_path, True) == str(model.resolve())


@pytest.mark.parametrize(
    ("model", "repo_id", "filename"),
    [
        ("small", "ggerganov/whisper.cpp", "ggml-small.bin"),
        (
            "large-v3-turbo-q5_0",
            "ggerganov/whisper.cpp",
            "ggml-large-v3-turbo-q5_0.bin",
        ),
        ("someone/repo/ggml-model.bin", "someone/repo", "ggml-model.bin"),
    ],
)
def test_model_names_map_to_hub_files(model, repo_id, filename, tmp_path) -> None:
    from wyoming_vulkan_whisper import whisper_cpp_handler

    with patch.object(whisper_cpp_handler, "hf_hub_download", return_value="x") as dl:
        whisper_cpp_handler.resolve_ggml_model(model, tmp_path, True)

    dl.assert_called_once_with(
        repo_id, filename, cache_dir=str(tmp_path), local_files_only=True
    )


@pytest.mark.parametrize("model", ["rhasspy/faster-whisper-base-int8", "not-a-model"])
def test_unknown_model_is_rejected(model, tmp_path) -> None:
    from wyoming_vulkan_whisper.whisper_cpp_handler import resolve_ggml_model

    with pytest.raises(ValueError, match="whisper.cpp model"):
        resolve_ggml_model(model, tmp_path, True)
