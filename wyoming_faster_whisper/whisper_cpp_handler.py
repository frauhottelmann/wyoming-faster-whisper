"""Code for Whisper transcription using whisper.cpp (via pywhispercpp).

whisper.cpp is the one Whisper runtime here with a Vulkan backend, which is what
makes it worth having next to faster-whisper: it reaches AMD and Intel GPUs -
including APU iGPUs that ROCm does not support - with nothing but the Mesa
driver. Built without Vulkan (the PyPI wheels) it runs on the CPU, so the same
code serves both.
"""

import logging
import threading
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from huggingface_hub import hf_hub_download
from pywhispercpp.constants import AVAILABLE_MODELS
from pywhispercpp.model import Model

from .const import Transcriber
from .device import whisper_cpp_gpu
from .languages import whisper_language

_LOGGER = logging.getLogger(__name__)

_RATE = 16000

# Where whisper.cpp publishes its ggml conversions of the OpenAI checkpoints.
GGML_REPO_ID = "ggerganov/whisper.cpp"


def resolve_ggml_model(
    model_id: str, cache_dir: Union[str, Path], local_files_only: bool
) -> str:
    """Return the path of a ggml model file, downloading it if needed.

    Accepts a path to a ggml ``.bin`` file, one of whisper.cpp's model names
    ("small", "large-v3-turbo-q5_0", ...), or ``<repo>/<file>.bin`` for a ggml
    file in any other HuggingFace repository.

    Downloads go through huggingface_hub rather than pywhispercpp's own
    downloader, so the files land in the same cache as every other backend's,
    and a cache miss under local_files_only raises an OSError
    (LocalEntryNotFoundError) - which is what the loader's cache-first retry
    expects.
    """
    path = Path(model_id)
    if path.is_file():
        return str(path.resolve())

    if model_id in AVAILABLE_MODELS:
        repo_id, filename = GGML_REPO_ID, f"ggml-{model_id}.bin"
    elif model_id.count("/") >= 2:
        owner, repo, filename = model_id.split("/", 2)
        repo_id = f"{owner}/{repo}"
    else:
        raise ValueError(
            f"Unknown whisper.cpp model '{model_id}'. Use one of "
            f"{', '.join(AVAILABLE_MODELS)}, a path to a ggml .bin file, or "
            "<owner>/<repo>/<file>.bin"
        )

    return hf_hub_download(
        repo_id,
        filename,
        cache_dir=str(cache_dir),
        local_files_only=local_files_only,
    )


class WhisperCppTranscriber(Transcriber):
    """Wrapper for a whisper.cpp model."""

    def __init__(
        self,
        model_id: str,
        cache_dir: Union[str, Path],
        local_files_only: bool = False,
        device: str = "cpu",
        cpu_threads: int = 4,
        beam_size: int = 5,
        task: Optional[str] = None,
    ) -> None:
        model_path = resolve_ggml_model(model_id, cache_dir, local_files_only)

        use_gpu, gpu_device = whisper_cpp_gpu(device)
        _LOGGER.debug(
            "Loading whisper.cpp model %s (use_gpu=%s, gpu_device=%s)",
            model_path,
            use_gpu,
            gpu_device,
        )

        params: Dict[str, Any] = {
            "n_threads": cpu_threads,
            "translate": task == "translate",
            "no_context": True,
            "print_progress": False,
            "print_realtime": False,
            "print_timestamps": False,
        }

        # The sampling strategy is fixed when the context is created, so the
        # beam size configured at startup decides it.
        sampling_strategy = 0
        if beam_size > 1:
            sampling_strategy = 1
            params["beam_search"] = {"beam_size": beam_size, "patience": -1.0}

        self.model = Model(
            model_path,
            params_sampling_strategy=sampling_strategy,
            # False leaves whisper.cpp's own log on stderr, which is where it
            # says which Vulkan device it found - or that it found none.
            redirect_whispercpp_logs_to=False,
            context_params={
                "use_gpu": use_gpu,
                "gpu_device": gpu_device,
                "flash_attn": use_gpu,
            },
            **params,
        )

        # A whisper.cpp context holds the decoder state of the transcription in
        # progress, so two clients must not run whisper_full on it at once.
        # transcribe() is called from worker threads (asyncio.to_thread).
        self._lock = threading.Lock()

    def transcribe(
        self,
        wav_path: Union[str, Path],
        language: Optional[str],
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> str:
        """Returns transcription for WAV file.

        WAV file must be 16Khz 16-bit mono audio.
        """
        wav_file: wave.Wave_read = wave.open(str(wav_path), "rb")
        with wav_file:
            assert wav_file.getframerate() == _RATE, "Sample rate must be 16Khz"
            assert wav_file.getsampwidth() == 2, "Width must be 16-bit (2 bytes)"
            assert wav_file.getnchannels() == 1, "Audio must be mono"
            audio_bytes = wav_file.readframes(wav_file.getnframes())

        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Parameters given to Model.transcribe persist into later calls, so
        # every one of them is set each time - an empty prompt included.
        with self._lock:
            segments = self.model.transcribe(
                audio,
                language=whisper_language(language) or "auto",
                initial_prompt=initial_prompt or "",
            )

        return " ".join(segment.text for segment in segments).strip()
