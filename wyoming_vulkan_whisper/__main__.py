#!/usr/bin/env python3
import argparse
import asyncio
import logging
import platform
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, List, Optional

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer, AsyncTcpServer

from . import __version__
from .const import AUTO_LANGUAGE, AUTO_MODEL, HASS_API_URL, base_language
from .dispatch_handler import DispatchEventHandler
from .env_args import ENV_PREFIX
from .env_args import parse_args as parse_args_with_env
from .models import ModelLoader, is_distil_whisper
from .vocabulary import DEFAULT_PROMPT_MAX_TOKENS

if TYPE_CHECKING:
    # Imported for typing only: aiohttp is an optional extra.
    from .name_cache import HassNameCache

_LOGGER = logging.getLogger(__name__)


def positive_float(value: str) -> float:
    """Parse a strictly positive floating-point command-line value."""
    number = float(value)
    if number <= 0:
        raise ValueError("must be greater than zero")
    return number


def get_parser() -> argparse.ArgumentParser:
    """Return the command-line parser.

    Every option here also has an environment variable, so an argument added
    below needs nothing extra to be settable from a container's environment.
    """
    parser = argparse.ArgumentParser(
        epilog=(
            "Every option can also be set from the environment: --some-option "
            f"reads {ENV_PREFIX}SOME_OPTION, or {ENV_PREFIX}SOME_OPTION_FILE "
            "to read the value out of a file (docker secrets). "
            "Command-line arguments win over both."
        )
    )
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    #
    parser.add_argument(
        "--zeroconf",
        nargs="?",
        const="vulkan-whisper",
        help="Enable discovery over zeroconf with optional name (default: vulkan-whisper)",
    )
    #
    parser.add_argument(
        "--model", default=AUTO_MODEL, help=f"Name of model to use (or {AUTO_MODEL})"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        action="append",
        help="Data directory to check for downloaded models",
    )
    parser.add_argument(
        "--download-dir",
        help="Directory to download models into (default: first data dir)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use for inference: vulkan, vulkan:N (the N-th Vulkan "
        "device), or cpu (default: cpu; the Docker image defaults to vulkan)",
    )
    parser.add_argument(
        "--language",
        default=AUTO_LANGUAGE,
        help=f"Default language to set for transcription (default: {AUTO_LANGUAGE})",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=0,
        help="Size of beam during decoding (0 for auto)",
    )
    parser.add_argument(
        "--cpu-threads",
        default=4,
        type=int,
        help="Number of CPU threads to use for inference (default: 4)",
    )
    parser.add_argument(
        "--initial-prompt",
        help="Optional text to provide as a prompt for the first window",
    )
    parser.add_argument(
        "--whisper-task",
        default="transcribe",
        help="Task for whisper to do (default: transcribe)",
        choices=["transcribe", "translate"],
    )
    parser.add_argument(
        "--vad-endpointing",
        type=positive_float,
        help="End a command and send its transcript after this many seconds of "
        "VAD-detected silence (default: wait for audio-stop)",
    )
    parser.add_argument(
        "--vad-clip",
        action="store_true",
        help="Use pysilero-vad to clip leading/trailing silence before "
        "transcription (default: disabled)",
    )
    parser.add_argument(
        "--vad-clip-threshold",
        type=float,
        default=0.5,
        help="Speech probability threshold for --vad-clip (default: 0.5)",
    )
    parser.add_argument(
        "--vad-clip-pad-ms",
        type=int,
        default=400,
        help="Context kept around speech for --vad-clip in ms (default: 400)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Never download a model: fail if it isn't already in --download-dir "
        "(cached models are loaded without a download regardless)",
    )
    # Home Assistant name biasing (extra: hass)
    parser.add_argument(
        "--hass-token",
        help="Long-lived access token for Home Assistant. Enables biasing toward "
        "the names of exposed entities, areas, and floors (extra: hass). Keep it "
        f"off the command line with {ENV_PREFIX}HASS_TOKEN_FILE",
    )
    parser.add_argument(
        "--hass-api",
        default=HASS_API_URL,
        help=f"URL of the Home Assistant API (default: {HASS_API_URL})",
    )
    parser.add_argument(
        "--hass-refresh-seconds",
        type=float,
        default=0.0,
        help="Minimum seconds between name refreshes (default: 0, refresh on "
        "every utterance)",
    )
    parser.add_argument(
        "--hass-prompt-max-tokens",
        type=int,
        default=DEFAULT_PROMPT_MAX_TOKENS,
        help="Token budget for names in the prompt "
        f"(default: {DEFAULT_PROMPT_MAX_TOKENS}, max useful: 223 for Whisper)",
    )
    parser.add_argument(
        "--hass-prompt-timeout",
        type=float,
        default=1.0,
        help="Seconds to wait for an unfinished name refresh before transcribing "
        "with the previous names (default: 1.0)",
    )
    #
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format", default=logging.BASIC_FORMAT, help="Format for log messages"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )

    return parser


async def main() -> None:
    """Main entry point."""
    args = parse_args_with_env(get_parser())

    if not args.download_dir:
        # Download to first data dir by default
        args.download_dir = args.data_dir[0]

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    machine = platform.machine().lower()
    is_arm = ("arm" in machine) or ("aarch" in machine)

    if args.beam_size <= 0:
        args.beam_size = 1 if is_arm else 5
        _LOGGER.debug("Beam size automatically selected: %s", args.beam_size)

    if args.language == AUTO_LANGUAGE:
        # Whisper does not understand auto
        args.language = None

    if args.model == AUTO_MODEL:
        args.model = None

    loader = ModelLoader(
        preferred_language=args.language,
        download_dir=args.download_dir,
        local_files_only=args.local_files_only,
        model=args.model,
        device=args.device,
        beam_size=args.beam_size,
        cpu_threads=args.cpu_threads,
        initial_prompt=args.initial_prompt,
        whisper_task=args.whisper_task,
        vad_clip=args.vad_clip,
        vad_clip_threshold=args.vad_clip_threshold,
        vad_clip_pad_ms=args.vad_clip_pad_ms,
    )

    model_name = loader.resolve_model()
    _warn_if_prompt_unsupported(args, model_name)

    # A ".en" model only knows English, so the advertised list follows the model.
    languages = sorted(loader.supported_languages())
    _warn_if_language_unsupported(args.language, languages)
    wyoming_info = build_info(
        model_name,
        languages,
        requires_external_vad=args.vad_endpointing is None,
    )
    _LOGGER.debug("Reporting %s language(s)", len(languages))

    # Load model
    _LOGGER.debug("Pre-loading transcriber")
    await loader.load_transcriber()

    names = await _load_names(args)

    server = AsyncServer.from_uri(args.uri)

    if args.zeroconf:
        if not isinstance(server, AsyncTcpServer):
            raise ValueError("Zeroconf requires tcp:// uri")

        from wyoming.zeroconf import HomeAssistantZeroconf

        tcp_server: AsyncTcpServer = server
        hass_zeroconf = HomeAssistantZeroconf(
            name=args.zeroconf, port=tcp_server.port, host=tcp_server.host
        )
        await hass_zeroconf.register_server()
        _LOGGER.debug("Zeroconf discovery enabled")

    _LOGGER.info("Ready")
    await server.run(
        partial(
            DispatchEventHandler,
            wyoming_info,
            loader,
            names,
            vad_endpointing=args.vad_endpointing,
        )
    )


def build_info(
    model_name: str,
    languages: List[str],
    *,
    requires_external_vad: bool = True,
) -> Info:
    """Build Wyoming service metadata.

    ``languages`` comes from ModelLoader.supported_languages(), which narrows
    Whisper's list to English for a ".en" model.
    """
    return Info(
        asr=[
            AsrProgram(
                name="whisper.cpp",
                description="Whisper transcription with whisper.cpp (Vulkan)",
                attribution=Attribution(
                    name="ggml-org",
                    url="https://github.com/ggml-org/whisper.cpp",
                ),
                installed=True,
                version=__version__,
                requires_external_vad=requires_external_vad,
                models=[
                    AsrModel(
                        name=model_name,
                        description=model_name,
                        attribution=Attribution(
                            name="OpenAI",
                            url="https://huggingface.co/ggerganov/whisper.cpp",
                        ),
                        installed=True,
                        languages=languages,
                        version=_pywhispercpp_version(),
                    )
                ],
            )
        ],
    )


def _pywhispercpp_version() -> Optional[str]:
    """The installed pywhispercpp version, read without loading whisper.cpp."""
    try:
        return version("pywhispercpp")
    except PackageNotFoundError:
        return None


# -----------------------------------------------------------------------------


def _warn_if_language_unsupported(
    language: Optional[str], languages: List[str]
) -> None:
    """Warn at startup when --language names something the model cannot do.

    Unvalidated, a typo or a language the model does not cover only shows up as
    every transcription being auto-detected (or, on an English-only model,
    silently decoded as the wrong language). A region-qualified tag is accepted:
    it is narrowed to its base language before transcription.
    """
    if not language:
        return

    if language in languages:
        return

    if base_language(language) in {base_language(code) for code in languages}:
        return

    _LOGGER.warning(
        "Language '%s' is not supported by the selected model (supported: %s). "
        "Transcription will fall back to auto-detection.",
        language,
        ", ".join(languages),
    )


def _warn_if_prompt_unsupported(args: argparse.Namespace, model: str) -> None:
    """Warn when a prompt is being sent to a model that cannot handle one.

    Distil-Whisper checkpoints degrade rather than benefit: the prompt truncates
    or derails the transcription instead of biasing it. The prompt is still sent
    -- this only warns -- because the damage is size-dependent and a very short
    --initial-prompt may be harmless.
    """
    if not (args.initial_prompt or args.hass_token):
        return

    if not is_distil_whisper(model):
        return

    _LOGGER.warning(
        "Model '%s' is a Distil-Whisper model, which is not compatible with "
        "prompting: --initial-prompt and --hass-token/--hass-api will degrade "
        "transcription instead of improving it. Use a standard Whisper model "
        "for name biasing.",
        model,
    )


async def _load_names(args: argparse.Namespace) -> Optional["HassNameCache"]:
    """Set up biasing toward Home Assistant's names, if a token was given."""
    if not args.hass_token:
        return None

    try:
        from .hass_api import HomeAssistant
        from .name_cache import HassNameCache
    except ImportError as exc:
        raise ImportError(
            "--hass-token requires the 'hass' extra: pip install -e '.[hass]'"
        ) from exc

    names = HassNameCache(
        HomeAssistant(args.hass_token, api_url=args.hass_api),
        prefix=args.initial_prompt,
        max_tokens=args.hass_prompt_max_tokens,
        refresh_seconds=args.hass_refresh_seconds,
        wait_timeout=args.hass_prompt_timeout,
    )

    # Fetch once now so the first utterance is biased too. Best effort: Home
    # Assistant may still be starting up, and that must not stop this server
    # from serving. Every utterance retries.
    if await names.refresh():
        _LOGGER.info("Biasing toward names from %s", args.hass_api)
    else:
        _LOGGER.warning(
            "No names loaded from %s yet; will retry on each utterance",
            args.hass_api,
        )

    return names


# -----------------------------------------------------------------------------


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
