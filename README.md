# MangaKoi AI — Manga Translation Worker

A **RunPod Serverless worker** that automatically translates manga pages through a 5-stage ML pipeline. Downloads source images from Cloudflare R2, processes them, and uploads the translated result back.

## Pipeline (5 Stages)

1. **Detection** (`pipeline/detect.py`) — RT-DETRv2 detects text bubbles and text-free regions
2. **OCR** (`pipeline/ocr.py`) — manga-ocr extracts Japanese text from detected regions
3. **Translation** (`pipeline/translate.py`) — Qwen3-14B (4-bit quantized by default) translates to target language
4. **Inpainting** (`pipeline/inpaint.py`) — LaMa removes original text bubbles
5. **Typesetting** (`pipeline/typeset.py`) — PIL renders translated text with manga-appropriate fonts

## Quick Start

### Prerequisites

- RunPod account with GPU pod/endpoint
- Cloudflare R2 bucket (or any S3-compatible storage)
- Environment variables configured in RunPod console

### Environment Variables (Required)

| Variable | Purpose |
|---|---|
| `BUCKET_ENDPOINT_URL` | Cloudflare R2 endpoint URL |
| `BUCKET_ACCESS_KEY_ID` | R2 API access key |
| `BUCKET_SECRET_ACCESS_KEY` | R2 API secret key |
| `BUCKET_NAME` | R2 bucket name |

### Optional Variables

| Variable | Default | Purpose |
|---|---|---|
| `HF_HOME` | `~/.cache/huggingface` | HF model cache path (set to `/runpod-volume/huggingface` for persistence) |
| `USE_4BIT_TRANSLATION` | `true` | 4-bit quantization for Qwen3-14B (~12GB VRAM vs ~28GB full) |
| `LOG_LEVEL` | `INFO` | DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `LOG_FORMAT` | `json` | `json` (structured) or `text` (human-readable) |
| `WEBHOOK_TIMEOUT_SECONDS` | `10` | Timeout for completion webhook POST |

### Deployment

```bash
# Build the Docker image
docker build -t mangakoi-ai .

# Push to your container registry and deploy on RunPod
```

## Input / Output

**Request:**
```json
{
  "input": {
    "inputObjectKey": "uploads/page_001.png",
    "targetLanguage": "en",
    "readingDirection": "rtl"
  }
}
```

**Response:**
```json
{
  "status": "done",
  "jobId": "abc-123-def",
  "outputObjectKey": "outputs/abc-123-def/translated.png",
  "elapsedSeconds": 45.2
}
```

## Architecture

Models are loaded **once at module level** (not inside the handler) so warm workers reuse them across jobs. The handler validates input, downloads the image from R2, runs the pipeline, uploads the result, and cleans up temp files — all with structured JSON logging per job.

```
handler.py → R2 download → run_full_pipeline() → R2 upload
                ↑                    ↓
          5-stage pipeline   Models (loaded once)
```

## VRAM Requirements

- **Qwen3-14B 4-bit (default):** ~12–14 GB
- **Qwen3-14B full precision:** ~28 GB
- **Other models:** ~8 GB combined

Set `USE_4BIT_TRANSLATION=false` if GPU has >24 GB VRAM and you want higher translation quality.

## Project Structure

```
├── handler.py              # RunPod entry point
├── Dockerfile              # Container build
├── pipeline/
│   ├── __init__.py         # Pipeline orchestration
│   ├── models.py           # Model loading
│   ├── detect.py           # Stage 1: Detection
│   ├── ocr.py              # Stage 2: OCR
│   ├── translate.py        # Stage 3: Translation
│   ├── inpaint.py          # Stage 4: Inpainting
│   └── typeset.py          # Stage 5: Typesetting
├── utils/
│   └── logger.py           # Structured logging with request IDs
├── .env.example            # Environment variable reference
└── requirements.txt        # Pinned dependencies
```

## Notes

- **Reading direction** (`rtl`/`ltr`) is accepted in the input schema but text-bubble reordering logic is not yet implemented — feature reserved for future development.
- All Python packages in `requirements.txt` are pinned to exact versions for reproducibility.
- This is a production serverless worker with no test suite; testing is done via RunPod console or `--test_input` local invocation.
