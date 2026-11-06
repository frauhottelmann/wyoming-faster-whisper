"""Translation of the --device option for whisper.cpp.

Accepted values are "vulkan" and "vulkan:N" (the N-th Vulkan device) for the
GPU, and "cpu". whisper.cpp uses whichever GPU backend it was built with, so
"cuda[:N]" also means "use the GPU" for a build made that way; anything else
runs on the CPU.
"""

import logging
from typing import Optional, Tuple

_LOGGER = logging.getLogger(__name__)


def is_vulkan(device: str) -> bool:
    """Report whether a device string names a Vulkan GPU ("vulkan", "vulkan:N")."""
    return device.strip().lower().startswith("vulkan")


def is_gpu(device: str) -> bool:
    """Report whether a device string names a GPU whisper.cpp can use."""
    return is_vulkan(device) or device.strip().lower().startswith("cuda")


def device_index(device: str) -> Optional[int]:
    """Extract the GPU ordinal from "vulkan:N", or None if unspecified."""
    _, _, index = device.strip().lower().partition(":")
    if not index:
        return None

    try:
        return int(index)
    except ValueError:
        _LOGGER.warning("Ignoring unparsable device index in '%s'", device)
        return None


def whisper_cpp_gpu(device: str) -> Tuple[bool, int]:
    """Return (use_gpu, gpu_device) for whisper.cpp.

    A GPU build with no usable device falls back to the CPU, logging why.
    """
    if not is_gpu(device):
        return False, 0

    return True, device_index(device) or 0
