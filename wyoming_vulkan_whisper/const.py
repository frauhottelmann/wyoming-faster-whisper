"""Constants."""

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

AUTO_LANGUAGE = "auto"
AUTO_MODEL = "auto"

# Where to look for Home Assistant when biasing toward its names.
HASS_API_URL = "http://homeassistant.local:8123/api"


# Home Assistant's own tag parser splits on either separator, so we do too.
_SEPARATORS = re.compile(r"[-_]")


def base_language(code: str) -> str:
    """Strip the region from a language tag and lower-case it.

    "en-US" -> "en". Home Assistant sends back the tag we advertised, but the
    --language option is typed by hand and other Wyoming clients send whatever
    they like.
    """
    return _SEPARATORS.split(code.lower(), maxsplit=1)[0]


class Transcriber(ABC):
    """Base class for transcribers."""

    @abstractmethod
    def transcribe(
        self,
        wav_path: Union[str, Path],
        language: Optional[str],
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> str:
        pass

    def count_prompt_tokens(self, text: str) -> Optional[int]:
        """Count the tokens ``text`` would use as an initial_prompt.

        Returns None when this backend has no tokenizer to ask, in which case
        callers fall back to an estimate. Used to fit as many entity names as
        possible into the prompt without crossing the model's context limit.
        """
        return None
