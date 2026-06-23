"""
Shared utility functions used across pipeline stages.
"""

import colorsys
import gc

import torch
from PIL import Image

from utils import get_logger

logger = get_logger(__name__)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def flush_vram():
    """Force-free GPU memory. Call between pipeline stages if VRAM is tight."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def vram_status(label: str = ""):
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    tag = f"[{label}] " if label else ""
    logger.info(
        f"VRAM {tag}— alloc: {alloc:.2f} GB  reserved: {reserved:.2f} GB  total: {total:.2f} GB"
    )


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def rgb_luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def rgb_saturation(rgb: tuple[int, int, int]) -> float:
    r, g, b = [x / 255.0 for x in rgb]
    _, s, _ = colorsys.rgb_to_hsv(r, g, b)
    return s


def is_achromatic(
    rgb: tuple[int, int, int], saturation_threshold: float = 0.15
) -> bool:
    """True if the color is a white/grey/black shade (no real color identity)."""
    return rgb_saturation(rgb) < saturation_threshold


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")
