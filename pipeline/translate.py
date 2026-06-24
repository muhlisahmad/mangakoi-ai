"""
Stage 3 — Translation.

Model: Qwen/Qwen3-14B
Non-thinking mode (enable_thinking=False) — manga dialogue doesn't need
chain-of-thought reasoning, and skipping it roughly halves latency.

All text on a page is translated in a single batched prompt so the model
can use cross-panel context (character names, pronoun continuity, tone).
"""

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils import get_logger

logger = get_logger(__name__)

TRANSLATION_MODEL_ID = "Qwen/Qwen3-14B"


def load_translator(device_map: str, use_4bit: bool, cache_dir: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_ID, cache_dir=cache_dir)

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            TRANSLATION_MODEL_ID,
            quantization_config=bnb_config,
            device_map=device_map,
            dtype=torch.bfloat16,
            cache_dir=cache_dir,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            TRANSLATION_MODEL_ID,
            device_map=device_map,
            dtype=torch.bfloat16,
            cache_dir=cache_dir,
        )

    model.eval()
    return model, tokenizer


def _build_prompt(texts: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    return (
        "You are a professional manga translator. "
        "Translate each numbered Japanese text into natural English. "
        "Keep the same numbering. "
        "You should use colloquial or slang or nsfw vocabulary if it makes the translation more accurate. "
        "Preserve tone, emotion, and speech style (formal/informal/yelling/whispering). "
        "Do NOT add notes or explanations — output only the numbered translations.\n\n"
        f"Japanese texts:\n{numbered}"
    )


def _parse_response(response: str, n: int) -> list[str]:
    translations = {}
    for match in re.finditer(r"^(\d+)[.)]\s*(.+)", response, re.MULTILINE):
        translations[int(match.group(1))] = match.group(2).strip()
    logger.debug(
        "Translation response parsed",
        extra={"expected": n, "received": len(translations)},
    )
    return [translations.get(i + 1, "[translation error]") for i in range(n)]


def run_translation(texts: list[str], model, tokenizer) -> list[str]:
    if not texts:
        return []

    prompt = _build_prompt(texts)
    logger.debug("Built translation prompt", extra={"prompt": prompt})
    text_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
    logger.debug("Input tokenized", extra={"input_tokens": len(inputs.input_ids[0])})

    logger.debug(
        "Starting model generation",
        extra={"temperature": 0.6, "top_k": 20, "top_p": 0.95, "max_new_tokens": 1024},
    )
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.6,
            top_k=20,
            top_p=0.95,
        )

    output_ids = generated[0][len(inputs.input_ids[0]) :]
    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    logger.debug(
        "Translation generation complete",
        extra={
            "input_tokens": len(inputs.input_ids[0]),
            "output_tokens": len(output_ids),
        },
    )
    return _parse_response(response, len(texts))


def translate_ocr_results(
    ocr_results: list[dict[str, object]], model, tokenizer
) -> list[dict[str, object]]:
    """Translate all non-empty OCR results from a page and re-attach translations."""
    logger.debug("Translating OCR results", extra={"ocr_count": len(ocr_results)})
    texts = [str(r["text"]) for r in ocr_results if r["text"]]
    if not texts:
        logger.warning("No text to translate", extra={"ocr_count": len(ocr_results)})
        return [{**r, "translation": ""} for r in ocr_results]

    translated = run_translation(texts, model, tokenizer)
    t_iter = iter(translated)

    result = []
    for r in ocr_results:
        if r["text"]:
            result.append({**r, "translation": next(t_iter)})
        else:
            result.append({**r, "translation": ""})
    logger.debug("Translation complete", extra={"results_with_translation": result})
    return result
