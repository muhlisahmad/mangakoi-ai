# AGENTS.md — Manga Translation Worker

## Quick Reference

**Project Type**: RunPod Serverless worker — processes manga pages through a 5-stage ML pipeline  
**Storage**: Cloudflare R2 (S3-compatible object storage)  
**Deploy Target**: Docker container on RunPod GPU hosts  
**Framework**: PyTorch + transformers; models load at module level, not in handler  
**Env Config**: Startup validation for required S3/R2 credentials; fail fast on missing secrets

---

## Architecture Overview

### 5-Stage Pipeline (Sequential)

1. **Detection** (`pipeline/detect.py`): Detect text bubbles & text-free regions → RT-DETRv2
2. **OCR** (`pipeline/ocr.py`): Extract Japanese text from regions → manga-ocr
3. **Translation** (`pipeline/translate.py`): Translate to target language → Qwen3-14B (4-bit quantization optional)
4. **Inpainting** (`pipeline/inpaint.py`): Remove original text bubbles → LaMa
5. **Typesetting** (`pipeline/typeset.py`): Re-render translated text with fonts → PIL

**Entry Flow**:  
`handler.py` (RunPod handler) → downloads image from R2 → `run_full_pipeline()` in `pipeline/__init__.py` → uploads result back to R2

---

## Handler & RunPod Best Practices

### Key Pattern: Models Load Outside Handler

```python
# handler.py, lines 62-68
MODELS = load_all_models()  # ← Module level, NOT inside handler()

def handler(job):
    # Warm workers reuse MODELS across jobs — no reload per request
    final_image = run_full_pipeline(image, MODELS, rtl=rtl)
    return result
```

**Why**: RunPod warm workers must load models once at startup. If you load inside `handler()`, every job reloads everything.

### Input Validation Schema

`handler.py:73–83` — Uses `runpod.serverless.utils.rp_validator.validate()` with this schema:

```python
INPUT_SCHEMA = {
    "inputObjectKey": {"type": str, "required": True},           # S3 key of source image
    "sourceLanguage": {"type": str, "required": False, "default": "ja"},
    "targetLanguage": {"type": str, "required": False, "default": "en"},
    "readingDirection": {
        "type": str,
        "required": False,
        "default": "rtl",
        "constraints": lambda x: x in ("rtl", "ltr"),
    },
}
```

**Note**: `readingDirection` is accepted but the text-bubble reordering logic is **commented out** in `pipeline/__init__.py:43` — feature incomplete, kept for future development.

### Cleanup & Request Tracking

```python
# handler.py:89, 151-155
with set_request_id(job_id):
    try:
        # Process job
    finally:
        clean()  # Reclaim temp disk space — RunPod best practice
```

---

## Environment Configuration

### Required Variables (Validated at Startup)

Must be set in RunPod console; worker **fails fast** if missing (`handler.py:42–50`):

| Variable                   | Purpose                   | Example                                   |
| -------------------------- | ------------------------- | ----------------------------------------- |
| `BUCKET_ENDPOINT_URL`      | Cloudflare R2 endpoint    | `https://abc123.r2.cloudflarestorage.com` |
| `BUCKET_ACCESS_KEY_ID`     | R2 API token (access key) | Your R2 API access key                    |
| `BUCKET_SECRET_ACCESS_KEY` | R2 API token (secret key) | Your R2 API secret key                    |
| `BUCKET_NAME`              | R2 bucket name            | `manga-translation-bucket`                |

### Optional Variables (with Defaults)

| Variable                  | Default                | Purpose                                                                                  |
| ------------------------- | ---------------------- | ---------------------------------------------------------------------------------------- |
| `HF_HOME`                 | `~/.cache/huggingface` | HuggingFace model cache — **set to `/runpod-volume/huggingface` for persistent caching** |
| `USE_4BIT_TRANSLATION`    | `true`                 | 4-bit quantization for Qwen3-14B; set `false` if GPU VRAM < 24GB                         |
| `WEBHOOK_TIMEOUT_SECONDS` | `10`                   | Timeout for webhook POST requests on job completion                                      |
| `LOG_LEVEL`               | `INFO`                 | DEBUG, INFO, WARNING, ERROR, CRITICAL                                                    |
| `LOG_FORMAT`              | `json`                 | `json` (structured) or `text` (human-readable)                                           |
| `LOG_DIR`                 | `/runpod-volume/logs`  | Persistent log directory on network volume                                               |
| `PYTHONUNBUFFERED`        | `1`                    | Unbuffered stdout/stderr for real-time console logs                                      |

**Startup Order Matters**: See `handler.py:29–68`:

1. Setup logging
2. Validate required S3/R2 env vars
3. Create S3 client with credentials
4. Load all models (this is heavy — happens once)

---

## Model Loading & VRAM Management

### Load Pattern

Models are instantiated at module import time via `pipeline/models.py:load_all_models()`:

```python
# Each model loaded with explicit device/quantization config:
load_detector(device=device, cache_dir=cache_dir)      # RT-DETRv2 (detection)
load_ocr(force_cpu=(device == "cpu"))                  # manga-ocr
load_translator(device_map=..., use_4bit=use_4bit)     # Qwen3-14B
load_inpainter()                                        # LaMa
```

**Cache Location**: Set `HF_HOME` env var to RunPod network volume mount for persistence across warm/cold starts.

### VRAM Tuning

- **Qwen3-14B 4-bit (default)**: ~12–14 GB (set `USE_4BIT_TRANSLATION=true`)
- **Qwen3-14B full precision**: ~28 GB (set `USE_4BIT_TRANSLATION=false`)
- **Other models**: ~8 GB combined

Utilities in `pipeline/utils.py`:

- `get_device()` — returns "cuda" or "cpu"
- `flush_vram()` — force GPU cache clear between stages
- `vram_status(label)` — log VRAM usage

---

## Logging & Request Tracking

### Architecture

- **Setup**: `utils/logger.py` provides `setup_logging()`, `get_logger()`, `set_request_id()`
- **Format**: JSON by default (structured logs for log aggregation); can switch to human-readable text
- **Request ID**: Tracked via `ContextVar` (thread-safe) — each job gets `job["id"]` from RunPod

### Usage Pattern

```python
# handler.py:29–34 (one-time setup)
logger = setup_logging(log_level="INFO", log_format="json")

# handler.py:89 (per-job context)
with set_request_id(job_id):
    logger.info("Processing job...")  # Request ID automatically included
```

### Persistent Logs

- Console logs: auto-captured by RunPod
- Persistent logs: written to `LOG_DIR` (default `/runpod-volume/logs`) if set
- Log filenames: ISO 8601 date format (`worker-YYYY-MM-DD.log`)

---

## Dockerfile & Deployment

### Key Build Decisions

```dockerfile
# Base: CUDA 12.8 + cuDNN 9 + PyTorch 2.11
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

# System deps required by pipeline models
RUN apt-get install -y fonts-comic-neue libgl1 libglib2.0-0
```

**Why these packages**:

- `fonts-comic-neue`: Typesetting stage needs this font (manga dialogue aesthetic)
- `libgl1`, `libglib2.0-0`: Required by opencv-python-headless at import time

### Layer Caching

1. **Layer 1**: Base image (CUDA)
2. **Layer 2**: `requirements.txt` (pip install) — cached; only rebuilds if requirements change
3. **Layer 3**: Application code (pipeline/, handler.py) — rebuilds on code changes
4. **Layer 4**: `simple-lama-inpainting` installed separately with `--no-deps` to avoid nested dependency conflicts

### Environment Defaults in Dockerfile

```dockerfile
ENV USE_4BIT_TRANSLATION="false"  # Default to full precision in build
ENV HF_HOME="/runpod-volume/hf_cache"
```

**Note**: Runtime vars (S3 credentials, webhook config) are set in RunPod console, not baked into image.

---

## Input & Output Flow

### Request Payload (from client)

```json
{
  "input": {
    "inputObjectKey": "uploads/page_001.png",
    "targetLanguage": "en",
    "readingDirection": "rtl"
  }
}
```

### Response Payload

RunPod wraps the handler's return value in the `"output"` field and adds top-level metadata. The `status` inside `output` is application-level; the top-level `status` reflects RunPod SDK's job state.

**Success — `/runsync` response:**

```json
{
  "delayTime": 824,
  "executionTime": 3391,
  "id": "sync-79164ff4-d212-44bc-9fe3-389e199a5c15",
  "output": {
    "status": "done",
    "outputObjectKey": "outputs/sync-79164ff4-d212-44bc-9fe3-389e199a5c15/translated.png",
    "elapsedSeconds": 45.2
  },
  "status": "COMPLETED"
}
```

**Success — `/run` then `/status` response:**

```json
{
  "delayTime": 31618,
  "executionTime": 1437,
  "id": "60902e6c-08a1-426e-9cb9-9eaec90f5e2b-u1",
  "output": {
    "status": "done",
    "outputObjectKey": "outputs/60902e6c-08a1-426e-9cb9-9eaec90f5e2b-u1/translated.png",
    "elapsedSeconds": 45.2
  },
  "status": "COMPLETED"
}
```

**Failure (caught exception — job still `COMPLETED` at RunPod level):**

```json
{
  "delayTime": 1200,
  "executionTime": 5000,
  "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "output": {
    "status": "failed",
    "error": "ConnectionError: R2 endpoint unreachable"
  },
  "status": "COMPLETED"
}
```

> **Note**: The handler's `try/except` catches all exceptions and returns a dict, so RunPod always reports `"status": "COMPLETED"` at the top level. Check `output.status` for the actual pipeline result. Uncaught exceptions propagate to RunPod SDK and set top-level `"status": "FAILED"`.

**Output Location**: `outputs/{job_id}/translated.png` in R2 bucket

---

## Development Notes

### Reading Direction & Bubble Reordering (Incomplete Feature)

The `readingDirection` parameter is accepted in the input schema, but the associated text-bubble reordering logic is **commented out**:

- `pipeline/detect.py:76–171` — Function `sort_detections_reading_order()` is commented
- `pipeline/__init__.py:43` — Call to reordering is commented with note about immaturity

**For now**: RTL/LTR affects no actual processing. Feature reserved for future development once algorithm matures.

### No Test Suite

This is a production serverless worker; no unit/integration tests included. Testing is done via RunPod console or local handler invocation with `--test_input` flag.

### Exact Versions

All Python packages in `requirements.txt` are **pinned to exact versions** — reproducibility across different RunPod hosts and over time.

---

## Common Pitfalls for Agents

1. **Loading models in handler()** — ❌ Don't. Load at module level so warm workers reuse them.
2. **Forgetting `clean()` in finally block** — ❌ Temp files accumulate → `DiskQuotaExceeded` over many jobs.
3. **Hardcoding S3 credentials** — ❌ Use environment variables; RunPod console only.
4. **Setting `HF_HOME` to non-persistent path** — ❌ Models re-download on cold start. Use `/runpod-volume/huggingface`.
5. **Ignoring input validation errors** — ❌ Validate early; return structured error response.
6. **Using `print()` for logging** — ❌ Use structured logging (`setup_logging()`) for request ID tracking and JSON output.
7. **Assuming `readingDirection` sorts bubbles** — ❌ That feature is commented out and immature.

---

## Key Files & Line References

| File                   | Purpose                                | Key Lines                                                |
| ---------------------- | -------------------------------------- | -------------------------------------------------------- |
| `handler.py`           | RunPod entry point, validation, S3 I/O | 42–50 (validation), 62–68 (model load), 86–155 (handler) |
| `pipeline/__init__.py` | 5-stage orchestration                  | 20–72 (run_full_pipeline)                                |
| `pipeline/models.py`   | Model loading at startup               | 25–62 (load_all_models)                                  |
| `utils/logger.py`      | Structured logging, request IDs        | ContextVar tracking for thread safety                    |
| `Dockerfile`           | Build optimizations, dependencies      | Layer caching strategy, system deps                      |
| `.env.example`         | Env var documentation                  | Full reference for all variables                         |

---

## RunPod Best Practices Applied

This worker follows these RunPod guidelines:

- ✓ **Handler Best Practices**: Models load outside handler; input validation before processing; cleanup in finally block
- ✓ **Input Validation**: Using `rp_validator.validate()` with schema
- ✓ **Environment Variables**: Startup validation, fail-fast on missing secrets
- ✓ **Logging**: Structured JSON logs with request ID tracking; optional persistent storage
- ✓ **Cleanup**: `rp_cleanup.clean()` reclaims disk space after job completes or fails

See [RunPod Serverless docs](https://docs.runpod.io/serverless) for details.
