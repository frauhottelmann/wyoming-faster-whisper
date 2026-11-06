"""Whisper's language codes, and normalizing a client's code onto them."""

import logging
from typing import Optional

from .const import base_language

_LOGGER = logging.getLogger(__name__)

# The language tokens of the multilingual Whisper checkpoints (large-v3 added
# "yue"), in Whisper's own order. Kept here rather than asked of whisper.cpp so
# the list is available without loading its native library.
WHISPER_LANGUAGES = frozenset(
    (
        "en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms "
        "cs ro da hu ta no th ur hr bg lt la mi ml cy sk te fa lv bn sr az sl kn "
        "et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc ka be "
        "tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln "
        "ha ba jw su yue"
    ).split()
)


def whisper_language(language: Optional[str]) -> Optional[str]:
    """Normalize a client's language code to one Whisper knows, else None.

    Home Assistant normally sends back one of the codes we advertised, but it
    falls through to the raw pipeline language when its matcher finds nothing
    (Irish and Cornish are Home Assistant languages Whisper has no token for) or
    when a pipeline has no stt_language stored, in which case a locale tag like
    "pt-BR" arrives verbatim. Strip the region if that helps; otherwise return
    None and let Whisper auto-detect, which is a far better outcome than an
    error.
    """
    if not language:
        return None

    code = language.lower()
    if code in WHISPER_LANGUAGES:
        return code

    base = base_language(code)
    if base in WHISPER_LANGUAGES:
        _LOGGER.debug("Language '%s' narrowed to '%s' for Whisper", language, base)
        return base

    _LOGGER.warning(
        "Language '%s' is not supported by Whisper; auto-detecting instead",
        language,
    )
    return None
