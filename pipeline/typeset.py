"""
Stage 5 — Typesetting.

Renders translated text back onto the cleaned (inpainted) image.
No GPU required — pure PIL/Pillow + OpenCV + scikit-learn (color analysis).

Color identity logic (see pipeline.utils for is_achromatic/rgb_luminance/
rgb_saturation):

  is_region_chromatic? (any meaningfully saturated pixels in the ORIGINAL
  region — distinguishes B&W manga from full-color manga)

    YES — full-color path:
      text_bubble:
        bubble background achromatic (white/grey)?
          → identity color = most-saturated k-means cluster from the
            original-vs-cleaned diff → used directly as TEXT fill
        bubble background chromatic (colored bubble tint)?
          → author already expressed identity via bubble tint;
            text fill just needs readability against that background
      text_free:
        → identity color = lightest k-means cluster (the outline stroke)
          → used as STROKE color
        → text fill chosen for contrast against the STROKE (not the
          background) — the stroke is the separation medium, exactly
          like the bubble is for text_bubble

    NO — B&W path:
      text_bubble: text fill contrasts with bubble background, no stroke
      text_free:   text fill contrasts with background; stroke = opposite
"""

import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans

from pipeline.utils import (is_achromatic, rgb_luminance, rgb_saturation,
                            rgb_to_hex)
from utils import get_logger

logger = get_logger(__name__)

MAX_FONT_SIZE = 32
MIN_FONT_SIZE = 8
FONT_PATH = "/usr/share/fonts/opentype/comic-neue/ComicNeue-Bold.otf"
STROKE_WIDTH_FREE = 3


# ── Color identity extraction ───────────────────────────────────────


def is_region_chromatic(
    original_image: Image.Image,
    box: list[int],
    saturation_threshold: float = 0.15,
    chromatic_pixel_ratio: float = 0.05,
) -> bool:
    """True if a meaningful fraction of the region's original pixels are
    chromatic — i.e. this is a full-color manga panel, not B&W."""
    x1, y1, x2, y2 = box
    crop = np.array(original_image.crop((x1, y1, x2, y2)).convert("RGB"))
    H, W = crop.shape[:2]
    total = H * W
    if total == 0:
        return False

    crop_f32 = crop.astype(np.float32) / 255.0
    cmax = np.max(crop_f32, axis=2)
    cmin = np.min(crop_f32, axis=2)
    delta = cmax - cmin
    sat = np.where(cmax > 0, delta / cmax, 0.0)

    chromatic_pixels = np.sum(sat > saturation_threshold)
    return bool((chromatic_pixels / total) >= chromatic_pixel_ratio)


def extract_identity_color(
    original_image: Image.Image,
    cleaned_image: Image.Image,
    box: list[int],
    min_text_pixels: int = 50,
    cluster_selection: str = "saturation",  # 'saturation' or 'luminance'
) -> tuple[int, int, int] | None:
    """
    Extract the artist's identity color for a text region by diffing the
    original image against the inpainted (cleaned) image.

    Pipeline:
      1. Grayscale absdiff between original and cleaned crop
      2. Otsu thresholding (auto-calibrated per crop, no magic numbers)
      3. Morphological closing — fills hollow glyph interiors
      4. Connected-component filtering — drops small noise blobs
      5. Isolate original pixels under the clean mask
      6. K-means (k=2) to separate the two dominant colors
      7. Pick the cluster per cluster_selection:
         'saturation' → most colorful cluster (identity = text fill,
                        used when bubble bg is achromatic)
         'luminance'  → lightest cluster (identity = outline stroke,
                        used for text_free)
    """
    x1, y1, x2, y2 = box
    orig_crop = np.array(original_image.crop((x1, y1, x2, y2)).convert("RGB"))
    clean_crop = np.array(cleaned_image.crop((x1, y1, x2, y2)).convert("RGB"))

    H, W = orig_crop.shape[:2]
    if H < 4 or W < 4:
        return None

    gray_orig = cv2.cvtColor(orig_crop, cv2.COLOR_RGB2GRAY)
    gray_clean = cv2.cvtColor(clean_crop, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(gray_orig, gray_clean)
    _, diff_mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        diff_mask, connectivity=8
    )
    clean_mask = np.zeros_like(diff_mask)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= 10:
            clean_mask[labels == label_id] = 255

    if clean_mask.sum() < min_text_pixels * 255:
        return None

    text_pixels = orig_crop[clean_mask > 0]
    if len(text_pixels) < 2:
        return None

    n_clusters = min(2, len(text_pixels))
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=5,  # pyright: ignore[reportArgumentType]
        random_state=0,
    )
    kmeans.fit(text_pixels.astype(np.float32))
    centers = kmeans.cluster_centers_

    if cluster_selection == "saturation":
        scores = [rgb_saturation(tuple(c)) for c in centers]
        best_idx = int(np.argmax(scores))
        if scores[best_idx] < 0.15:
            return None
    else:  # 'luminance'
        scores = [rgb_luminance(tuple(c)) for c in centers]
        best_idx = int(np.argmax(scores))

    return tuple(
        int(v) for v in centers[best_idx].astype(np.uint8)
    )  # pyright: ignore[reportReturnType]


def get_adaptive_colors(
    cleaned_image: Image.Image,
    box: list[int],
    label: str,
    original_image: Image.Image,
) -> tuple[str, str | None, int]:
    """Determine (text_color_hex, stroke_color_hex_or_None, stroke_width)."""
    x1, y1, x2, y2 = box
    crop = cleaned_image.crop((x1, y1, x2, y2)).convert("RGB")
    avg_color = crop.resize((1, 1), Image.Resampling.LANCZOS).getpixel((0, 0))
    bg_rgb = avg_color[:3]  # pyright: ignore[reportIndexIssue, reportOptionalSubscript]
    bg_lum = rgb_luminance(bg_rgb)  # pyright: ignore[reportArgumentType]

    is_chromatic = original_image is not None and is_region_chromatic(
        original_image, box
    )
    logger.debug(
        "Color path decision",
        extra={
            "is_chromatic": is_chromatic,
            "label": label,
            "box": box,
            "bg_luminance": round(bg_lum, 1),
        },
    )

    # ── FULL-COLOR PATH ─────────────────────────────────────────
    if is_chromatic:
        if label == "text_bubble":
            if is_achromatic(bg_rgb):  # pyright: ignore[reportArgumentType]
                identity = extract_identity_color(
                    original_image, cleaned_image, box, cluster_selection="saturation"
                )
                if identity is not None:
                    logger.debug(
                        "Color result — full-color manga with chromatic bubble background. Set identity color for text_color",
                        extra={
                            "text_color": rgb_to_hex(identity),
                            "stroke_color": None,
                            "stroke_width": 0,
                        },
                    )
                    return rgb_to_hex(identity), None, 0
                text_color = "#1a1a1a" if bg_lum >= 128 else "#FFFFFF"
                logger.debug(
                    "Color result — full-color manga with chromatic text bubble background. Can't get identity color for text_color, fallback to B/W coloring",
                    extra={
                        "text_color": text_color,
                        "stroke_color": None,
                        "stroke_width": 0,
                    },
                )
                return text_color, None, 0
            else:
                text_color = "#1a1a1a" if bg_lum >= 128 else "#FFFFFF"
                logger.debug(
                    "Color result — full-color manga with colored text bubble background. Set text_color based on contrast",
                    extra={
                        "text_color": text_color,
                        "stroke_color": None,
                        "stroke_width": 0,
                    },
                )
                return text_color, None, 0

        else:  # text_free
            identity = extract_identity_color(
                original_image, cleaned_image, box, cluster_selection="luminance"
            )
            if identity is not None:
                id_lum = rgb_luminance(identity)
                text_color = "#1a1a1a" if id_lum >= 128 else "#FFFFFF"
                logger.debug(
                    "Color result — full-color manga, setting text stroke color as identity color.",
                    extra={
                        "text_color": text_color,
                        "stroke_color": rgb_to_hex(identity),
                        "stroke_width": STROKE_WIDTH_FREE,
                    },
                )
                return text_color, rgb_to_hex(identity), STROKE_WIDTH_FREE
            text_color = "#1a1a1a" if bg_lum >= 128 else "#FFFFFF"
            stroke_color = "#FFFFFF" if text_color == "#1a1a1a" else "#1a1a1a"
            logger.debug(
                "Color result — full-color manga. Can't get identity color for text_free, fallback to B/W coloring",
                extra={
                    "text_color": text_color,
                    "stroke_color": stroke_color,
                    "stroke_width": STROKE_WIDTH_FREE,
                },
            )
            return text_color, stroke_color, STROKE_WIDTH_FREE

    # ── B&W PATH ─────────────────────────────────────────────────
    if label == "text_bubble":
        text_color = "#1a1a1a" if bg_lum >= 128 else "#FFFFFF"
        logger.debug(
            "Color result — B&W text_bubble",
            extra={"text_color": text_color, "stroke_color": None, "stroke_width": 0},
        )
        return text_color, None, 0
    else:  # text_free
        text_color = "#1a1a1a" if bg_lum >= 128 else "#FFFFFF"
        stroke_color = "#FFFFFF" if text_color == "#1a1a1a" else "#1a1a1a"
        logger.debug(
            "Color result — B&W text_free",
            extra={
                "text_color": text_color,
                "stroke_color": stroke_color,
                "stroke_width": STROKE_WIDTH_FREE,
            },
        )
        return text_color, stroke_color, STROKE_WIDTH_FREE


# ── Text fitting and rendering ──────────────────────────────────────


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except (IOError, OSError):
        logger.warning("Font fallback to default", extra={"path": FONT_PATH})
        return ImageFont.load_default()


def _fit_text(
    draw: ImageDraw.ImageDraw, text: str, box_w: int, box_h: int
) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, list[str]]:
    """Find the largest font size where wrapped text fits the box.
    break_long_words/break_on_hyphens are disabled so wrapping never
    splits a word mid-character — narrow vertical-derived bubbles would
    otherwise produce broken words like 'presiden' / 't?'."""
    logger.debug(
        "Fitting text", extra={"text_len": len(text), "box_size": f"{box_w}x{box_h}"}
    )
    for size in range(MAX_FONT_SIZE, MIN_FONT_SIZE - 1, -1):
        font = _load_font(size)
        chars_per_line = max(1, int(box_w / (size * 0.6)))
        lines = textwrap.wrap(
            text,
            width=chars_per_line,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [text]
        line_h = size + 4
        total_h = line_h * len(lines)
        max_line_w = max(draw.textlength(l, font=font) for l in lines)
        if max_line_w <= box_w * 0.95 and total_h <= box_h * 0.90:
            logger.debug(
                "Text fit found", extra={"font_size": size, "lines": len(lines)}
            )
            return font, lines

    logger.debug(
        "Text fit exhausted — using min size",
        extra={"font_size": MIN_FONT_SIZE},
    )
    font = _load_font(MIN_FONT_SIZE)
    chars = max(1, int(box_w / (MIN_FONT_SIZE * 0.6)))
    lines = textwrap.wrap(
        text, width=chars, break_long_words=False, break_on_hyphens=False
    ) or [text]
    return font, lines


def _draw_text_in_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: list[int],
    text_color: str,
    stroke_color: str | None = None,
    stroke_width: int = 0,
):
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    if box_w < 20 or box_h < 20:
        logger.debug("Skipped tiny box", extra={"box": box, "size": f"{box_w}x{box_h}"})
        return

    font, lines = _fit_text(draw, text, box_w, box_h)
    font_size = (
        font.size  # pyright: ignore[reportAttributeAccessIssue]
        if hasattr(font, "size")
        else MIN_FONT_SIZE
    )
    line_h = font_size + 4
    total_h = line_h * len(lines)
    y_start = y1 + (box_h - total_h) // 2
    logger.debug(
        "Rendering text",
        extra={
            "font_size": font_size,
            "lines": len(lines),
            "text_color": text_color,
            "stroke_color": stroke_color,
            "stroke_width": stroke_width
        },
    )

    for i, line in enumerate(lines):
        line_w = draw.textlength(line, font=font)
        x_centered = x1 + (box_w - line_w) // 2
        draw.text(
            (x_centered, y_start + i * line_h),
            line,
            fill=text_color,
            font=font,
            stroke_width=stroke_width if stroke_color else 0,
            stroke_fill=stroke_color,
        )


def run_typesetting(
    cleaned_image: Image.Image,
    ocr_results: list[dict[str, object]],
    original_image: Image.Image,
) -> Image.Image:
    logger.debug("Starting typesetting", extra={"region_counts": len(ocr_results)})
    output = cleaned_image.copy()
    draw = ImageDraw.Draw(output)

    bubble_count, free_count = 0, 0
    for region in ocr_results:
        translation = str(region.get("translation", "")).strip()
        if not translation or translation == "[translation error]":
            logger.debug(
                "Skipped region — no translation",
                extra={"label": region.get("label"), "box": region.get("box")},
            )
            continue
        label = region.get("label")
        if label not in ("text_bubble", "text_free"):
            continue

        text_color, stroke_color, stroke_width = get_adaptive_colors(
            cleaned_image,
            region["box"],  # pyright: ignore[reportArgumentType]
            label=label,
            original_image=original_image,
        )
        _draw_text_in_box(
            draw,
            translation,
            region["box"],  # pyright: ignore[reportArgumentType]
            text_color,
            stroke_color,
            stroke_width,
        )

        if label == "text_bubble":
            bubble_count += 1
        else:
            free_count += 1

    logger.debug(
        "Typesetting complete",
        extra={"text_bubble_count": bubble_count, "free_text_count": free_count},
    )
    return output
