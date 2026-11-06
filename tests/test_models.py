"""Tests for model selection, cache-first loading, and reported languages.

None of these load a model or need whisper.cpp's native library.
"""

import logging

import pytest

from wyoming_vulkan_whisper.__main__ import _warn_if_language_unsupported
from wyoming_vulkan_whisper.languages import WHISPER_LANGUAGES
from wyoming_vulkan_whisper.models import (
    ModelLoader,
    guess_model,
    is_distil_whisper,
    is_english_only_model,
)


def _loader(model=None, device="vulkan", local_files_only=False) -> ModelLoader:
    return ModelLoader(
        preferred_language="en",
        download_dir="/data",
        local_files_only=local_files_only,
        model=model,
        device=device,
        beam_size=5,
        cpu_threads=4,
        initial_prompt=None,
    )


# --- default model ---------------------------------------------------------


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
    assert guess_model(is_arm, gpu=gpu) == expected


@pytest.mark.parametrize(
    ("device", "expected"), [("vulkan", "small"), ("vulkan:1", "small")]
)
def test_vulkan_device_picks_the_gpu_default(device, expected) -> None:
    assert _loader(device=device).resolve_model() == expected


def test_explicit_model_wins() -> None:
    assert _loader(model="large-v3-turbo-q5_0").resolve_model() == (
        "large-v3-turbo-q5_0"
    )


def test_no_default_model_is_distilled() -> None:
    # A guessed model must never be one that biasing cannot work with.
    for is_arm in (False, True):
        for gpu in (False, True):
            assert not is_distil_whisper(guess_model(is_arm, gpu=gpu))


# --- Distil-Whisper detection ---------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "distil-small.en",
        "distil-whisper/distil-large-v3/ggml-distil-large-v3.bin",
        "DISTIL-SMALL.EN",  # case-insensitive
    ],
)
def test_distil_models_are_detected(model) -> None:
    assert is_distil_whisper(model)


@pytest.mark.parametrize("model", [None, "small", "small.en", "large-v3-turbo"])
def test_standard_models_are_not_detected(model) -> None:
    assert not is_distil_whisper(model)


# --- reported languages ----------------------------------------------------


def test_multilingual_model_reports_every_whisper_language() -> None:
    assert _loader().supported_languages() == set(WHISPER_LANGUAGES)
    assert _loader(model="large-v3-turbo").supported_languages() == set(
        WHISPER_LANGUAGES
    )


@pytest.mark.parametrize(
    "model",
    [
        "base.en",
        "small.en-q5_1",
        "/models/ggml-base.en.bin",
        "someone/repo/ggml-small.en.bin",
    ],
)
def test_english_only_model_reports_only_english(model) -> None:
    assert is_english_only_model(model)
    assert _loader(model=model).supported_languages() == {"en"}


@pytest.mark.parametrize(
    "model", [None, "small", "large-v3", "/models/ggml-medium.bin"]
)
def test_multilingual_models_are_not_english_only(model) -> None:
    assert not is_english_only_model(model)


# --- cache-first loading --------------------------------------------------


def _record_attempts(loader: ModelLoader, fail_cached: bool) -> list:
    """Replace model construction with a recorder of local_files_only values."""
    attempts: list = []

    def build(model, local_files_only):
        attempts.append(local_files_only)
        if local_files_only and fail_cached:
            # What huggingface_hub raises for a cache miss.
            raise FileNotFoundError("not cached")

        return f"transcriber(local_files_only={local_files_only})"

    loader._build_transcriber = build  # type: ignore[method-assign]
    return attempts


@pytest.mark.asyncio
async def test_cached_model_is_loaded_without_the_hub() -> None:
    # The common case: the model is on disk, so nothing should reach the network
    # even though --local-files-only was not passed.
    loader = _loader()
    attempts = _record_attempts(loader, fail_cached=False)

    assert await loader.load_transcriber() == "transcriber(local_files_only=True)"
    assert attempts == [True]


@pytest.mark.asyncio
async def test_missing_model_falls_back_to_downloading() -> None:
    loader = _loader()
    attempts = _record_attempts(loader, fail_cached=True)

    assert await loader.load_transcriber() == "transcriber(local_files_only=False)"
    assert attempts == [True, False]


@pytest.mark.asyncio
async def test_local_files_only_does_not_fall_back() -> None:
    # Explicitly asking for offline means a missing model is an error, not a
    # download.
    loader = _loader(local_files_only=True)
    attempts = _record_attempts(loader, fail_cached=True)

    with pytest.raises(FileNotFoundError):
        await loader.load_transcriber()

    assert attempts == [True]


@pytest.mark.asyncio
async def test_transcriber_is_only_built_once() -> None:
    loader = _loader()
    attempts = _record_attempts(loader, fail_cached=True)

    first = await loader.load_transcriber("en")
    assert await loader.load_transcriber("de") is first
    assert attempts == [True, False]


# --- --language validation -------------------------------------------------


@pytest.mark.parametrize("language", [None, "en", "en-US", "EN"])
def test_supported_language_does_not_warn(language, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        _warn_if_language_unsupported(language, ["en"])

    assert not caplog.records


@pytest.mark.parametrize("language", ["de", "xyz", "de-DE"])
def test_unsupported_language_warns(language, caplog) -> None:
    # An English-only model: without this, a wrong --language only shows up as
    # every transcription being decoded as English.
    with caplog.at_level(logging.WARNING):
        _warn_if_language_unsupported(language, ["en"])

    assert len(caplog.records) == 1
    assert language in caplog.records[0].getMessage()
