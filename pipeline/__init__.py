"""
manga-saas-worker pipeline package.

run_full_pipeline() orchestrates all 5 stages in sequence for a single
manga page. This is the function the RunPod handler calls.
"""

from PIL import Image

from pipeline.detect import run_detection
from pipeline.inpaint import run_inpainting
from pipeline.ocr import run_ocr
from pipeline.translate import translate_ocr_results
from pipeline.typeset import run_typesetting
from utils import get_logger

logger = get_logger(__name__)


def run_full_pipeline(
    image: Image.Image, models: dict[object, object], rtl: bool = True
) -> Image.Image:
    """
    Run the complete 5-stage translation pipeline on a single manga page.

    Parameters
    ----------
    image : PIL.Image
        The source manga page (RGB).
    models : dict
        Output of pipeline.models.load_all_models().
    rtl : bool
        True for right-to-left (Japanese manga, default).
        False for left-to-right (Western comics, Korean webtoons).
    """
    device = models["device"]

    # ── Stage 1: Detection ──────────────────────────────────────
    logger.info("Stage 1/5 — Detection")
    all_detections = run_detection(
        image, models["detector_model"], models["detector_processor"], str(device)
    )
    # all_detections = sort_detections_reading_order(all_detections, image, rtl=rtl)

    text_bubble_boxes = [d for d in all_detections if d["label"] == "text_bubble"]
    text_free_boxes = [d for d in all_detections if d["label"] == "text_free"]
    text_regions = text_bubble_boxes + text_free_boxes

    # ── Stage 2: OCR ─────────────────────────────────────────────
    logger.info(f"Stage 2/5 — OCR ({len(text_regions)} regions)")
    ocr_results = run_ocr(image, text_regions, models["ocr_model"])
    ocr_with_text = [r for r in ocr_results if r["text"]]

    # ── Stage 3: Translation ────────────────────────────────────
    logger.info(f"Stage 3/5 — Translation ({len(ocr_with_text)} regions)")
    translated_results = translate_ocr_results(
        ocr_with_text, models["translator_model"], models["translator_tokenizer"]
    )

    # ── Stage 4: Inpainting ─────────────────────────────────────
    logger.info("Stage 4/5 — Inpainting (LaMa)")
    cleaned_image = run_inpainting(
        image, text_bubble_boxes, text_free_boxes, models["lama_model"]
    )

    # ── Stage 5: Typesetting ────────────────────────────────────
    logger.info("Stage 5/5 — Typesetting")
    final_image = run_typesetting(
        cleaned_image, translated_results, original_image=image
    )

    return final_image
