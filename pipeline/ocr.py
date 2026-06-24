"""
Stage 2 — OCR.

Model: kha-white/manga-ocr-base
Vision Encoder Decoder (ViT encoder + autoregressive decoder).
Reads one cropped text region at a time — not the full page.
"""

from typing import Any

from manga_ocr import MangaOcr
from PIL import Image

from utils import get_logger

logger = get_logger(__name__)

OCR_MODEL_ID = "kha-white/manga-ocr-base"
MIN_BOX_SIZE = 10


def load_ocr(force_cpu: bool = False) -> MangaOcr:
    return MangaOcr(pretrained_model_name_or_path=OCR_MODEL_ID, force_cpu=force_cpu)


def run_ocr(
    image: Image.Image,
    text_regions: list[dict[Any, Any]],
    ocr_model,
    min_box_size: int = MIN_BOX_SIZE,
) -> list[dict[str, object]]:
    logger.debug(
        "Starting OCR",
        extra={"text_regions": len(text_regions), "min_box_size": min_box_size},
    )
    results = []
    for region in text_regions:
        x1, y1, x2, y2 = region["box"]
        w, h = x2 - x1, y2 - y1
        if w < min_box_size or h < min_box_size:
            logger.debug(
                "Skipped small region",
                extra={"box": [x1, y1, x2, y2], "size": f"{w}x{h}"},
            )
            results.append({**region, "text": ""})
            continue
        crop = image.crop((x1, y1, x2, y2))
        text = ocr_model(crop).strip()
        results.append({**region, "text": text})
    logger.debug(
        "OCR complete", extra={"results_with_text": [r for r in results if r["text"]]}
    )
    return results
