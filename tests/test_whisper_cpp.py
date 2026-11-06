"""Tests for the whisper.cpp backend that need no model and no GPU."""

from pathlib import Path
from unittest.mock import patch

import pytest

from wyoming_faster_whisper.const import SttLibrary
from wyoming_faster_whisper.device import is_vulkan, whisper_cpp_gpu
from wyoming_faster_whisper.models import (
    WHISPER_LANGUAGES,
    ModelLoader,
    guess_model,
    guess_stt_library,
)

# --- --device ---------------------------------------------------------------


@pytest.mark.parametrize("device", ["vulkan", "vulkan:1", "VULKAN", " vulkan "])
def test_vulkan_is_recognized(device) -> None:
    assert is_vulkan(device)


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0", "auto", ""])
def test_everything_else_is_not_vulkan(device) -> None:
    assert not is_vulkan(device)


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


# --- backend selection -------------------------------------------------------

_NONE_AVAILABLE = dict(
    has_transformers=False,
    has_sherpa=False,
    has_onnx_asr=False,
    has_funasr=False,
    has_qwen3_asr=False,
)


def test_explicit_whisper_cpp_kept_when_available() -> None:
    assert (
        guess_stt_library(
            SttLibrary.WHISPER_CPP, None, "de", has_whisper_cpp=True, **_NONE_AVAILABLE
        )
        == SttLibrary.WHISPER_CPP
    )


def test_explicit_whisper_cpp_falls_back_when_missing() -> None:
    assert (
        guess_stt_library(
            SttLibrary.WHISPER_CPP, None, "de", has_whisper_cpp=False, **_NONE_AVAILABLE
        )
        == SttLibrary.FASTER_WHISPER
    )


def test_auto_never_selects_whisper_cpp() -> None:
    # It runs the same checkpoints as faster-whisper, so it is opt-in (the
    # Vulkan image opts in through WYO_WHISPER_STT_LIBRARY).
    for language in ("en", "de", "ru", "zh", None):
        assert guess_stt_library(
            SttLibrary.AUTO, None, language, has_whisper_cpp=True, **_NONE_AVAILABLE
        ) != (SttLibrary.WHISPER_CPP)


@pytest.mark.parametrize(
    ("gpu", "is_arm", "expected"),
    [
        (True, False, "small"),
        (True, True, "small"),
        (False, False, "base"),
        (False, True, "tiny"),
    ],
)
def test_default_model(gpu, is_arm, expected) -> None:
    for language in ("en", "de", None):
        assert (
            guess_model(SttLibrary.WHISPER_CPP, language, is_arm, gpu=gpu) == expected
        )


def _loader(device="vulkan", model=None) -> ModelLoader:
    return ModelLoader(
        preferred_stt_library=SttLibrary.WHISPER_CPP,
        preferred_language=None,
        download_dir="/data",
        local_files_only=False,
        model=model,
        compute_type="default",
        device=device,
        beam_size=5,
        cpu_threads=4,
        initial_prompt=None,
        vad_parameters=None,
    )


def test_reports_whisper_languages() -> None:
    loader = _loader()
    with patch.object(
        ModelLoader,
        "_available_backends",
        return_value={**_NONE_AVAILABLE, "has_whisper_cpp": True},
    ):
        assert loader.supported_languages() == set(WHISPER_LANGUAGES)


def test_english_only_model_reports_only_english() -> None:
    loader = _loader(model="base.en")
    with patch.object(
        ModelLoader,
        "_available_backends",
        return_value={**_NONE_AVAILABLE, "has_whisper_cpp": True},
    ):
        assert loader.supported_languages() == {"en"}


# --- model resolution (needs pywhispercpp, but no download) --------------------


def test_model_path_is_used_as_is(tmp_path: Path) -> None:
    pytest.importorskip("pywhispercpp")
    from wyoming_faster_whisper.whisper_cpp_handler import resolve_ggml_model

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
    pytest.importorskip("pywhispercpp")
    from wyoming_faster_whisper import whisper_cpp_handler

    with patch.object(whisper_cpp_handler, "hf_hub_download", return_value="x") as dl:
        whisper_cpp_handler.resolve_ggml_model(model, tmp_path, True)

    dl.assert_called_once_with(
        repo_id, filename, cache_dir=str(tmp_path), local_files_only=True
    )


@pytest.mark.parametrize("model", ["rhasspy/faster-whisper-base-int8", "not-a-model"])
def test_unknown_model_is_rejected(model, tmp_path) -> None:
    pytest.importorskip("pywhispercpp")
    from wyoming_faster_whisper.whisper_cpp_handler import resolve_ggml_model

    with pytest.raises(ValueError, match="whisper.cpp model"):
        resolve_ggml_model(model, tmp_path, True)
