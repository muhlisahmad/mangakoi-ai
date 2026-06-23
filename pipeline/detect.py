"""
Stage 1 — Text & bubble detection.

Model: ogkalu/comic-text-and-bubble-detector (RT-DETRv2 r50vd)
Classes: 0=bubble, 1=text_bubble, 2=text_free

Also includes panel-aware reading-order sorting so OCR/translation/
typesetting see text in the order a human would read the page —
critical for pronoun and dialogue-flow accuracy downstream.
"""

import torch
from PIL import Image
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

from utils import get_logger

logger = get_logger(__name__)

DETECTOR_MODEL_ID = "ogkalu/comic-text-and-bubble-detector"
ID2LABEL = {0: "bubble", 1: "text_bubble", 2: "text_free"}
DETECTION_THRESHOLD = 0.5


def load_detector(
    device: str, cache_dir: str | None = None
) -> tuple[RTDetrV2ForObjectDetection, RTDetrImageProcessor]:
    processor = RTDetrImageProcessor.from_pretrained(
        DETECTOR_MODEL_ID, cache_dir=cache_dir
    )
    model = RTDetrV2ForObjectDetection.from_pretrained(
        DETECTOR_MODEL_ID,
        dtype=torch.bfloat16,
        device_map=device,
        cache_dir=cache_dir,
    ).eval()
    return model, processor


def run_detection(
    image: Image.Image,
    model,
    processor,
    device: str,
    threshold: float = DETECTION_THRESHOLD,
) -> list[dict[object, object]]:
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device).to(torch.bfloat16) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([(image.height, image.width)]),
        threshold=threshold,
    )

    detections = []
    for score, label_id, box in zip(
        results[0]["scores"], results[0]["labels"], results[0]["boxes"]
    ):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        detections.append(
            {
                "label": ID2LABEL[label_id.item()],
                "score": round(score.item(), 4),
                "box": [x1, y1, x2, y2],
            }
        )
    return detections


# ── Panel-aware reading-order sorting ───────────────────────────────


# def find_panel_cuts(
#     image: Image.Image, axis: int, min_gap_ratio: float = 0.01, edge_margin: int = 5
# ) -> list[int]:
#     """
#     Find panel boundary positions by projecting the image onto one axis
#     and locating runs of near-empty content (panel gutters/borders).
#
#     axis=0 → horizontal cuts (row separators)
#     axis=1 → vertical cuts (column separators)
#     """
#     gray = np.array(image.convert("L"))
#     inverted = 255 - gray
#     projection = inverted.sum(axis=axis).astype(np.float32)
#     size = len(projection)
#     projection /= projection.max() + 1e-6
#
#     threshold = 0.05
#     is_gap = projection < threshold
#     min_gap_width = max(2, int(size * min_gap_ratio))
#
#     cuts = []
#     in_gap, gap_start = False, 0
#     for i in range(edge_margin, size - edge_margin):
#         if is_gap[i] and not in_gap:
#             in_gap, gap_start = True, i
#         elif not is_gap[i] and in_gap:
#             in_gap = False
#             gap_width = i - gap_start
#             if gap_width >= min_gap_width:
#                 cuts.append((gap_start + i) // 2)
#     return cuts


# def assign_boxes_to_panels(
#     boxes: list[list[int]], h_cuts: list[int], v_cuts: list[int]
# ) -> list[tuple[int, int]]:
#     row_boundaries = [0] + h_cuts + [10**7]
#     col_boundaries = [0] + v_cuts + [10**7]
#
#     assignments = []
#     for box in boxes:
#         x1, y1, x2, y2 = box
#         cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#
#         row = 0
#         for r in range(len(row_boundaries) - 1):
#             if row_boundaries[r] <= cy < row_boundaries[r + 1]:
#                 row = r
#                 break
#
#         col = 0
#         for c in range(len(col_boundaries) - 1):
#             if col_boundaries[c] <= cx < col_boundaries[c + 1]:
#                 col = c
#                 break
#
#         assignments.append((row, col))
#     return assignments


# def sort_detections_reading_order(
#     detections: list[dict[object, object]], image: Image.Image, rtl: bool = True
# ) -> list[dict]:
#     """
#     Sort detections into manga reading order:
#     1. Find panel boundaries via projection analysis
#     2. Assign each detection to a (row, col) panel cell
#     3. Sort panels top-to-bottom, then right-to-left (RTL) or left-to-right
#     4. Within each panel, sort top-to-bottom then right-to-left/left-to-right
#     """
#     if not detections:
#         return detections
#
#     boxes = [d["box"] for d in detections]
#     h_cuts = find_panel_cuts(image, axis=1)
#     v_cuts = find_panel_cuts(image, axis=0)
#     panel_assignments = assign_boxes_to_panels(boxes, h_cuts, v_cuts)
#
#     indexed = [
#         (det, row, col) for det, (row, col) in zip(detections, panel_assignments)
#     ]
#
#     col_sign = -1 if rtl else 1
#     x_sign = -1 if rtl else 1
#
#     def sort_key(item):
#         det, panel_row, panel_col = item
#         x1, y1, x2, y2 = det["box"]
#         cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#         return (panel_row, col_sign * panel_col, cy, x_sign * cx)
#
#     indexed.sort(key=sort_key)
#     return [det for det, _, _ in indexed]
