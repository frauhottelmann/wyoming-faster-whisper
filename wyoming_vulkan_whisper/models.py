"""Model selection, loading, and transcription."""

import asyncio
import logging
import platform
from pathlib import Path
from typing import Dict, Optional, Set, Union

from .const import Transcriber
from .device import is_gpu
from .languages import WHISPER_LANGUAGES

_LOGGER = logging.getLogger(__name__)


class ModelLoader:
    """Load the whisper.cpp transcriber and run transcriptions on it."""

    def __init__(
        self,
        preferred_language: Optional[str],
        download_dir: Union[str, Path],
        local_files_only: bool,
        model: Optional[str],
        device: str,
        beam_size: int,
        cpu_threads: int,
        initial_prompt: Optional[str],
        whisper_task: Optional[str] = None,
        vad_clip: bool = False,
        vad_clip_threshold: float = 0.5,
        vad_clip_pad_ms: int = 400,
    ) -> None:
        self.preferred_language = preferred_language

        self.download_dir = Path(download_dir)
        self.local_files_only = local_files_only

        self.model = model
        self.device = device

        self.beam_size = beam_size
        self.cpu_threads = cpu_threads
        self.initial_prompt = initial_prompt
        self.whisper_task = whisper_task
        self.vad_clip = vad_clip
        self.vad_clip_threshold = vad_clip_threshold
        self.vad_clip_pad_ms = vad_clip_pad_ms

        self._transcriber: Dict[str, Transcriber] = {}
        self._transcriber_lock = asyncio.Lock()

    def resolve_model(self) -> str:
        """Return the model to load: --model, or a default for the hardware."""
        if self.model is not None:
            return self.model

        machine = platform.machine().lower()
        is_arm = ("arm" in machine) or ("aarch" in machine)
        return guess_model(is_arm, gpu=is_gpu(self.device))

    def supported_languages(self) -> Set[str]:
        """Report the language codes the configured model can transcribe."""
        if is_english_only_model(self.model):
            return {"en"}

        return set(WHISPER_LANGUAGES)

    async def load_transcriber(self, language: Optional[str] = None) -> Transcriber:
        """Load the transcriber, or get it from the cache.

        ``language`` is accepted for the event handler's sake: one Whisper model
        serves every language, so it does not change what is loaded.
        """
        del language
        model = self.resolve_model()

        async with self._transcriber_lock:
            transcriber = self._transcriber.get(model)
            if transcriber is not None:
                return transcriber

            _LOGGER.debug("Loading model '%s' (device=%s)", model, self.device)
            transcriber = self._load_cache_first(model)
            self._transcriber[model] = transcriber

        return transcriber

    def _load_cache_first(self, model: str) -> Transcriber:
        """Build a transcriber, preferring the cache over the network.

        The hub check is what breaks when there is no internet, not the model
        load: everything a cached model needs is already on disk. On a network
        that blackholes outbound traffic (a Docker bridge marked `internal`) that
        check stalls for the full TCP timeout on every start rather than failing
        fast, which boot-loops the container. So try the cache first and only
        reach for the network when a file is genuinely missing.

        --local-files-only stays strict: no fallback, so a model that isn't
        cached is an error instead of a surprise download.
        """
        if self.local_files_only:
            return self._build_transcriber(model, local_files_only=True)

        try:
            return self._build_transcriber(model, local_files_only=True)
        except OSError as exc:
            # huggingface_hub reports a cache miss as LocalEntryNotFoundError, a
            # FileNotFoundError subclass. A corrupt cache lands here too, and the
            # retry below surfaces it as the download failing.
            _LOGGER.debug("Model '%s' is not cached (%s), downloading", model, exc)

        return self._build_transcriber(model, local_files_only=False)

    def _build_transcriber(self, model: str, local_files_only: bool) -> Transcriber:
        """Construct the whisper.cpp transcriber."""
        # Imported here so the rest of the server (and its tests) does not need
        # whisper.cpp's native library loaded.
        from .whisper_cpp_handler import WhisperCppTranscriber  # noqa: F811

        return WhisperCppTranscriber(
            model,
            cache_dir=self.download_dir,
            local_files_only=local_files_only,
            device=self.device,
            cpu_threads=self.cpu_threads,
            beam_size=self.beam_size,
            task=self.whisper_task,
        )


def is_distil_whisper(model: Optional[str]) -> bool:
    """Report whether a model id names a Distil-Whisper checkpoint.

    Distil-Whisper was distilled without previous-text conditioning, so prompt
    context never helps it. `distil-small.en` is actively damaged: a 52-token
    prompt pushes avg_logprob below -1.0 and yields truncated or looping output
    on audio it transcribes perfectly unprompted. `distil-large-v3` is instead
    inert -- stable at every prompt size, but with no biasing effect either.
    Nothing in a model file identifies a distilled checkpoint, so the model id is
    the only signal available.
    """
    return (model is not None) and ("distil" in model.lower())


def is_english_only_model(model: Optional[str]) -> bool:
    """Report whether a model id names an English-only Whisper checkpoint.

    Whisper publishes ".en" variants (tiny.en, base.en, small.en-q5_1, ...)
    whose tokenizer has no language tokens at all, so they transcribe English
    whatever language is requested. Reporting the full multilingual list for one
    would let Home Assistant route any pipeline to it.
    """
    if model is None:
        return False

    name = model.lower().split("/")[-1]
    if name.endswith(".bin"):
        name = name[: -len(".bin")]

    return name.endswith(".en") or (".en-" in name)


def guess_model(is_arm: bool, gpu: bool = False) -> str:
    """Pick a default whisper.cpp model.

    On a GPU, one size up from the CPU default: small is where Whisper starts
    being accurate outside English, and even an iGPU runs it faster than the
    CPU runs base.
    """
    if gpu:
        return "small"

    if is_arm:
        return "tiny"

    return "base"
