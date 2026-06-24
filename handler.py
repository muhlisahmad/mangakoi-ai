"""
RunPod Serverless handler — entry point for the manga translation worker.

Follows RunPod's documented best practices:
  - Models loaded OUTSIDE the handler (module level) so warm workers
    reuse them across jobs without reloading.
  - Input validated with runpod.serverless.utils.rp_validator.validate
    before any processing begins.
  - Structured logging with a per-job request_id via the logging module.
  - Temporary files cleaned up via rp_cleanup.clean() in a finally block.
  - All configuration read from environment variables with sane defaults,
    required secrets validated at startup (fail fast, not mid-job).
"""

import io
import os
import time

import boto3
import runpod
from PIL import Image
from runpod.serverless.utils.rp_cleanup import clean
from runpod.serverless.utils.rp_validator import validate

from pipeline import run_full_pipeline
from pipeline.models import load_all_models
from utils import set_request_id, setup_logging

# ── Logging setup ────────────────────────────────────────────────────
# Per RunPod's write-logs guidance: write to stdout/stderr via the
# standard `logging` module — RunPod automatically captures and displays
# these in the console. We tag every line with the job's request_id.

logger = setup_logging(log_level="INFO", log_format="json")


# ── Environment variable validation ──────────────────────────────────
# Per RunPod's environment-variables best practices: validate required
# secrets at startup so the worker fails fast and clearly instead of
# crashing mid-job with a confusing error.

REQUIRED_ENV_VARS = [
    "BUCKET_ENDPOINT_URL",
    "BUCKET_ACCESS_KEY_ID",
    "BUCKET_SECRET_ACCESS_KEY",
    "BUCKET_NAME",
]
_missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
if _missing:
    raise ValueError(f"Missing required environment variables: {', '.join(_missing)}")

BUCKET_NAME = os.environ["BUCKET_NAME"]
WEBHOOK_TIMEOUT_SECONDS = int(os.environ.get("WEBHOOK_TIMEOUT_SECONDS", "10"))

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["BUCKET_ENDPOINT_URL"],
    aws_access_key_id=os.environ["BUCKET_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["BUCKET_SECRET_ACCESS_KEY"],
)

# ── Load all models ONCE, outside the handler ────────────────────────
# This is the single most important RunPod best practice for this
# worker: heavy model loading happens here, at import time, so a warm
# worker processing job #2, #3, #N never reloads anything.
logger.info("Worker starting — loading models...")
MODELS = load_all_models()
logger.info("Worker ready.")


# ── Input validation schema ──────────────────────────────────────────

INPUT_SCHEMA = {
    "inputObjectKey": {"type": str, "required": True},
    "sourceLanguage": {"type": str, "required": False, "default": "ja"},
    "targetLanguage": {"type": str, "required": False, "default": "en"},
    "readingDirection": {
        "type": str,
        "required": False,
        "default": "rtl",
        "constraints": lambda x: x in ("rtl", "ltr"),
    },
}


def handler(job):
    job_id = job.get("id", "unknown")

    with set_request_id(job_id):
        try:
            # ── Validate input ───────────────────────────────────────
            validated = validate(job["input"], INPUT_SCHEMA)
            if "errors" in validated:
                logger.warning(f"Input validation failed: {validated['errors']}")
                return {"status": "failed", "error": validated["errors"]}

            v = validated["validated_input"]
            input_key = v[
                "inputObjectKey"
            ]  # pyright: ignore[reportArgumentType, reportCallIssue]
            rtl = (
                v[
                    "readingDirection"
                ]  # pyright: ignore[reportCallIssue, reportArgumentType]
                == "rtl"
            )

            logger.info("Processing job", extra={"input_key": input_key, "rtl": rtl})
            start_time = time.time()

            # ── Download source image from R2 ────────────────────────
            logger.debug("Downloading source from R2", extra={"key": input_key})
            response = s3.get_object(Bucket=BUCKET_NAME, Key=input_key)
            image = Image.open(io.BytesIO(response["Body"].read())).convert("RGB")
            logger.debug(
                "Source image loaded",
                extra={"image_size": f"{image.width}x{image.height}"},
            )

            # ── Run the 5-stage pipeline ──────────────────────────────
            logger.debug("Pipeline execution starting")
            final_image = run_full_pipeline(image, MODELS, rtl=rtl)

            # ── Upload result to R2 ───────────────────────────────────
            output_key = f"outputs/{job_id}/translated.png"
            logger.info("Uploading result to R2", extra={"key": output_key})
            buf = io.BytesIO()
            final_image.save(buf, format="PNG")
            _ = buf.seek(0)
            s3.put_object(
                Bucket=BUCKET_NAME, Key=output_key, Body=buf, ContentType="image/png"
            )

            elapsed = time.time() - start_time
            logger.info(
                "Job completed",
                extra={
                    "elapsed_seconds": round(elapsed, 1),
                    "output_key": output_key,
                },
            )

            result = {
                "status": "done",
                "outputObjectKey": output_key,
                "elapsedSeconds": round(elapsed, 1),
            }
            return result

        except Exception as e:
            logger.exception(f"Job failed: {e}")
            error_result = {
                "status": "failed",
                "error": str(e),
            }
            return error_result

        finally:
            # Per RunPod's cleanup best practices: always reclaim temp disk
            # space, even on failure, to avoid DiskQuotaExceeded over time
            # across many jobs on the same warm worker.
            clean()


runpod.serverless.start({"handler": handler})
