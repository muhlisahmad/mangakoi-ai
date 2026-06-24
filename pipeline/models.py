"""
Loads all pipeline models once, at worker startup.

Per RunPod's handler best practices, model loading must happen OUTSIDE
the handler function — at module import time — so warm workers reuse
already-loaded models across jobs instead of reloading per request.

Models are read from HF_HOME (pointed at the RunPod Network Volume by
the Dockerfile / endpoint env vars), so a warm OR cold-started worker
loads weights from local disk instead of re-downloading from HuggingFace.
"""

import os

from pipeline.detect import load_detector
from pipeline.inpaint import load_inpainter
from pipeline.ocr import load_ocr
from pipeline.translate import load_translator
from pipeline.utils import get_device
from utils import get_logger

logger = get_logger(__name__)


def load_all_models() -> dict[object, object]:
    device = get_device()
    cache_dir = os.environ.get("HF_HOME", None)  # set to the network volume mount path
    use_4bit = os.environ.get("USE_4BIT_TRANSLATION", "true").lower() == "true"

    logger.info(
        "Loading models",
        extra={"device": device, "use_4bit": use_4bit, "cache_dir": cache_dir},
    )

    logger.info("Loading detector (RT-DETRv2)...")
    try:
        detector_model, detector_processor = load_detector(
            device=device, cache_dir=cache_dir if cache_dir else None
        )
    except Exception:
        logger.critical("Failed to load detector model", exc_info=True)
        raise

    logger.info("Loading OCR (manga-ocr-base)...")
    try:
        ocr_model = load_ocr(force_cpu=(device == "cpu"))
    except Exception:
        logger.critical("Failed to load OCR model", exc_info=True)
        raise

    logger.info("Loading translator (Qwen3-14B)...")
    try:
        translator_model, translator_tokenizer = load_translator(
            device_map="auto" if device == "cuda" else "cpu",
            use_4bit=use_4bit,
            cache_dir=cache_dir if cache_dir else None,
        )
    except Exception:
        logger.critical("Failed to load translator model", exc_info=True)
        raise

    logger.info("Loading inpainter (LaMa)...")
    try:
        lama_model = load_inpainter()
    except Exception:
        logger.critical("Failed to load inpainter model", exc_info=True)
        raise

    logger.info("All models loaded successfully.")

    return {
        "device": device,
        "detector_model": detector_model,
        "detector_processor": detector_processor,
        "ocr_model": ocr_model,
        "translator_model": translator_model,
        "translator_tokenizer": translator_tokenizer,
        "lama_model": lama_model,
    }
