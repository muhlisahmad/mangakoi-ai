"""
Stage 4 — Inpainting.

Model: LaMa (Large Mask) via simple-lama-inpainting
Fourier-convolution architecture — resolution-robust, no fixed input
size, so the full-resolution manga page is processed with no resizing
and no quality loss.

Both text_bubble and text_free regions are inpainted by LaMa in a
single unified mask. Bubble shapes/tints, narration boxes, and SFX
backgrounds are all reconstructed; typesetting then renders translated
text back on top with adaptive coloring (see pipeline/typeset.py).
"""

from PIL import Image, ImageDraw
from simple_lama_inpainting import SimpleLama

from utils import get_logger

logger = get_logger(__name__)

BOX_PADDING = 4


def load_inpainter():
    return SimpleLama()


def _make_mask(
    image_size: tuple[int, int], boxes: list[list[int]], padding: int = BOX_PADDING
) -> Image.Image:
    W, H = image_size
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle(
            [
                max(0, x1 - padding),
                max(0, y1 - padding),
                min(W, x2 + padding),
                min(H, y2 + padding),
            ],
            fill=255,
        )
    return mask


def run_inpainting(
    image: Image.Image,
    text_bubble_detections: list[dict[object, object]],
    text_free_detections: list[dict[object, object]],
    lama_model,
) -> Image.Image:
    W, H = image.size
    all_boxes = [d["box"] for d in text_bubble_detections] + [
        d["box"] for d in text_free_detections
    ]
    if not all_boxes:
        return image

    mask = _make_mask((W, H), all_boxes)  # pyright: ignore[reportArgumentType]
    result = lama_model(image, mask)
    return result
