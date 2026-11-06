"""Event handler for clients of the server."""

import asyncio
import logging
import os
import tempfile
import time
import wave
from typing import TYPE_CHECKING, Optional

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .const import AUTO_LANGUAGE, Transcriber
from .endpointing import SileroEndpointDetector
from .models import ModelLoader
from .vad import clip_wav_to_speech

if TYPE_CHECKING:
    # Imported for typing only: the name cache pulls in aiohttp, which is an
    # optional extra.
    from .name_cache import HassNameCache

_LOGGER = logging.getLogger(__name__)


class DispatchEventHandler(AsyncEventHandler):
    """Collects a client's audio and answers with its transcript."""

    def __init__(
        self,
        wyoming_info: Info,
        loader: ModelLoader,
        names: Optional["HassNameCache"],
        *args,
        vad_endpointing: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.wyoming_info_event = wyoming_info.event()
        if vad_endpointing is not None and vad_endpointing <= 0:
            raise ValueError("vad_endpointing must be greater than zero")

        self._loader = loader
        self._names = names
        self._vad_endpointing = vad_endpointing
        self._endpoint_detector: Optional[SileroEndpointDetector] = None
        self._finished = False
        self._transcriber: Optional[Transcriber] = None
        self._transcriber_future: Optional[asyncio.Future] = None
        self._language: Optional[str] = None

        self._wav_dir = tempfile.TemporaryDirectory()
        self._wav_path = os.path.join(self._wav_dir.name, "speech.wav")
        self._clipped_wav_path = os.path.join(self._wav_dir.name, "clipped.wav")
        self._wav_file: Optional[wave.Wave_write] = None

        self._audio_converter = AudioChunkConverter(rate=16000, width=2, channels=1)

        self._got_audio = False

        # Whether this utterance already asked for a name refresh.
        self._refreshed_names = False

    async def handle_event(self, event: Event) -> bool:
        if AudioStart.is_type(event.type):
            self._finished = False
            self._reset_endpoint_detector()

            # Start refreshing Home Assistant's names now. The speaker is still
            # talking, so the fetch has the length of their command to finish in
            # and costs no latency at all.
            self._refresh_names()
            return True

        if AudioChunk.is_type(event.type):
            if self._finished:
                return True

            chunk = self._audio_converter.convert(AudioChunk.from_event(event))
            self._got_audio = True

            # Safety net for clients that send audio without AudioStart. Calls
            # after the first are cheap: a refresh already in flight is not
            # started again.
            self._refresh_names()

            if (self._transcriber is None) and (self._transcriber_future is None):
                # Load the transcriber in the background.
                # Hopefully it's ready by the time the audio stops.
                self._transcriber_future = asyncio.create_task(
                    self._loader.load_transcriber(self._language)
                )

            self._ensure_wav_file()
            assert self._wav_file is not None
            self._wav_file.writeframes(chunk.audio)

            if self._vad_endpointing is not None:
                if self._endpoint_detector is None:
                    # Safety net for clients that send chunks without AudioStart.
                    self._reset_endpoint_detector()

                assert self._endpoint_detector is not None
                if not self._endpoint_detector.process(chunk.audio):
                    _LOGGER.debug(
                        "Voice command ended after %.3f seconds of VAD silence",
                        self._vad_endpointing,
                    )
                    await self._finish_utterance()

            return True

        if AudioStop.is_type(event.type):
            _LOGGER.debug("Audio stopped")
            await self._finish_utterance()
            self._reset()
            return False

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            language = transcribe.language
            if language == AUTO_LANGUAGE:
                # A client can ask for detection by name; every backend spells
                # that as "no language". Passing "auto" through would make
                # whisper_language warn on each utterance.
                language = None

            self._language = language or self._loader.preferred_language
            _LOGGER.debug("Language set to %s", self._language)

            return True

        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        return True

    async def _finish_utterance(self) -> None:
        """Transcribe and answer the current stream exactly once."""
        if self._finished:
            return

        self._finished = True
        start_time = time.monotonic()

        # No audio was received before the stream ended.
        if not self._got_audio:
            _LOGGER.warning("Finishing utterance with no audio data")
            await self.write_event(Transcript(text="").event())
            return

        # Get the transcriber that was loading in the background.
        if self._transcriber is None:
            if self._transcriber_future is None:
                _LOGGER.warning("No transcriber available")
                await self.write_event(Transcript(text="").event())
                return
            self._transcriber = await self._transcriber_future

        assert self._wav_file is not None
        self._wav_file.close()
        self._wav_file = None

        # With --vad-clip, trim leading/trailing silence before transcription.
        wav_path = self._wav_path
        if self._loader.vad_clip and await asyncio.to_thread(
            clip_wav_to_speech,
            self._wav_path,
            self._clipped_wav_path,
            self._loader.vad_clip_threshold,
            self._loader.vad_clip_pad_ms,
        ):
            wav_path = self._clipped_wav_path

        # Do transcription in a separate thread
        text = await asyncio.to_thread(
            self._transcriber.transcribe,
            wav_path,
            self._language,
            beam_size=self._loader.beam_size,
            initial_prompt=await self._initial_prompt(),
        )

        end_time = time.monotonic()
        _LOGGER.info(text)

        await self.write_event(Transcript(text=text).event())
        _LOGGER.debug("Completed request in %s second(s)", end_time - start_time)

    def _reset_endpoint_detector(self) -> None:
        """Create or reset endpointing for a new normalized audio stream."""
        if self._vad_endpointing is None:
            return

        if self._endpoint_detector is None:
            self._endpoint_detector = SileroEndpointDetector(self._vad_endpointing)
        self._endpoint_detector.reset()

    def _refresh_names(self) -> None:
        """Kick off one background refresh of Home Assistant's names per utterance.

        Called from both AudioStart and every AudioChunk, so it has to latch: a
        finished fetch is immediately due again (--hass-refresh-seconds defaults
        to 0), and without the latch every chunk would start another one.
        """
        if (self._names is None) or self._refreshed_names:
            return

        self._refreshed_names = True
        self._names.schedule_refresh()

    async def _initial_prompt(self) -> Optional[str]:
        """The prompt to bias this utterance with.

        Without --hass-token this is just the configured --initial-prompt; with
        it, that prefix plus as many of Home Assistant's names as fit.
        """
        if self._names is None:
            return self._loader.initial_prompt

        prompt = await self._names.initial_prompt(self._transcriber)
        _LOGGER.debug("Initial prompt: %s", prompt)
        return prompt

    def _ensure_wav_file(self) -> None:
        """Open the temp WAV file if not already open."""
        if self._wav_file is None:
            self._wav_file = wave.open(self._wav_path, "wb")
            # Audio is normalized to this format by the converter.
            self._wav_file.setframerate(16000)
            self._wav_file.setsampwidth(2)
            self._wav_file.setnchannels(1)

    def _reset(self) -> None:
        """Reset per-request state."""
        if self._wav_file is not None:
            self._wav_file.close()
            self._wav_file = None

        self._language = None
        self._transcriber = None
        self._transcriber_future = None
        self._finished = False
        self._got_audio = False
        self._refreshed_names = False
