"""Gradio application for cross-variety sarcasm detection.

The application uses a shared 4-bit Gemma-2-2B backbone with three
variety-specific LoRA adapters trained for British, Australian, and
Indian English.
"""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from typing import Any

import gradio as gr
import torch
from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

LOGGER = logging.getLogger(__name__)

BASE_MODEL = os.getenv("BASE_MODEL", "google/gemma-2-2b")
HF_USER = os.getenv("HF_USER", "Mohammadeeu20")
MAX_LENGTH = 384

ADAPTER_REPOS = {
    "British (en-UK)": f"{HF_USER}/besstie-sarcasm-gemma2b-lora-en-uk",
    "Australian (en-AU)": f"{HF_USER}/besstie-sarcasm-gemma2b-lora-en-au",
    "Indian (en-IN)": f"{HF_USER}/besstie-sarcasm-gemma2b-lora-en-in",
}

ADAPTER_NAMES = {
    "British (en-UK)": "en-UK",
    "Australian (en-AU)": "en-AU",
    "Indian (en-IN)": "en-IN",
}

EXAMPLES = [
    [
        "Oh fantastic, another rail strike. Just what I needed this Monday morning.",
        "British (en-UK)",
    ],
    [
        "Yeah nah, the LNP really nailed it with this budget. Brilliant stuff.",
        "Australian (en-AU)",
    ],
    [
        "Wah, the new traffic rules are working so beautifully. Absolutely no jams now.",
        "Indian (en-IN)",
    ],
    [
        "I genuinely enjoyed the film — the cinematography was stunning.",
        "British (en-UK)",
    ],
    [
        "The match yesterday was a cracker, both teams played really well.",
        "Australian (en-AU)",
    ],
    [
        "The food at the new restaurant is quite good, would recommend.",
        "Indian (en-IN)",
    ],
]


def require_cuda() -> None:
    """Raise a clear error when CUDA is unavailable."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This application requires an NVIDIA CUDA GPU because the "
            "Gemma model is loaded using 4-bit bitsandbytes quantisation."
        )


@lru_cache(maxsize=1)
def load_resources() -> tuple[Any, PeftModel]:
    """Load the tokenizer, shared model, and all LoRA adapters once."""

    require_cuda()

    LOGGER.info("Loading tokenizer: %s", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    start_time = time.perf_counter()

    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    LOGGER.info(
        "Base model loaded in %.1f seconds.",
        time.perf_counter() - start_time,
    )

    model: PeftModel | None = None

    for variety_label, repository_id in ADAPTER_REPOS.items():
        adapter_name = ADAPTER_NAMES[variety_label]
        adapter_start = time.perf_counter()

        if model is None:
            model = PeftModel.from_pretrained(
                base_model,
                repository_id,
                adapter_name=adapter_name,
            )
        else:
            model.load_adapter(
                repository_id,
                adapter_name=adapter_name,
            )

        LOGGER.info(
            "Loaded %s adapter from %s in %.1f seconds.",
            variety_label,
            repository_id,
            time.perf_counter() - adapter_start,
        )

    if model is None:
        raise RuntimeError("No LoRA adapters were loaded.")

    model.eval()

    LOGGER.info(
        "Application ready. CUDA memory allocated: %.2f GB.",
        torch.cuda.memory_allocated() / 1e9,
    )

    return tokenizer, model


def predict(text: str, variety_label: str) -> tuple[str, float, str]:
    """Classify text as sarcastic or non-sarcastic."""

    cleaned_text = text.strip() if text else ""

    if not cleaned_text:
        return "—", 0.0, "Please enter some text."

    if variety_label not in ADAPTER_NAMES:
        return "—", 0.0, f"Unknown English variety: {variety_label}"

    tokenizer, model = load_resources()
    adapter_name = ADAPTER_NAMES[variety_label]
    model.set_adapter(adapter_name)

    input_device = model.get_input_embeddings().weight.device
    start_time = time.perf_counter()

    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(input_device)

    with torch.inference_mode():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits.float(),
        dim=-1,
    ).cpu().numpy()[0]

    probability_not_sarcastic = float(probabilities[0])
    probability_sarcastic = float(probabilities[1])

    is_sarcastic = probability_sarcastic >= 0.5
    prediction = "Sarcastic" if is_sarcastic else "Not sarcastic"

    confidence = (
        probability_sarcastic
        if is_sarcastic
        else probability_not_sarcastic
    )

    latency_ms = (time.perf_counter() - start_time) * 1000

    details = (
        f"Variety: **{variety_label}** "
        f"(adapter: `{adapter_name}`) · "
        f"P(sarcastic) = {probability_sarcastic:.3f} · "
        f"P(not sarcastic) = {probability_not_sarcastic:.3f} · "
        f"latency = {latency_ms:.0f} ms"
    )

    return prediction, confidence, details


def build_demo() -> gr.Blocks:
    """Construct the Gradio interface."""

    with gr.Blocks(
        title="BESSTIE Sarcasm Detector",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
# BESSTIE Sarcasm Detector

Detect sarcasm in **British, Australian, and Indian English** using
variety-specific LoRA adapters on Gemma-2-2B.

Select the English variety closest to the text's origin. Adapter
switching happens in milliseconds because each small LoRA adapter is
loaded on top of one shared base model.
"""
        )

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Text to classify",
                    placeholder="Type or paste a sentence…",
                    lines=4,
                )
                variety_input = gr.Dropdown(
                    choices=list(ADAPTER_NAMES),
                    value="British (en-UK)",
                    label="English variety",
                )
                predict_button = gr.Button(
                    "Detect sarcasm",
                    variant="primary",
                )

            with gr.Column(scale=1):
                prediction_output = gr.Label(label="Prediction")
                confidence_output = gr.Slider(
                    label="Confidence",
                    minimum=0,
                    maximum=1,
                    value=0,
                    interactive=False,
                )
                details_output = gr.Markdown()

        gr.Examples(
            examples=EXAMPLES,
            inputs=[text_input, variety_input],
            label="Try these examples",
        )

        gr.Markdown(
            f"""
**About:** The system uses `{BASE_MODEL}` with one of three LoRA
adapters fine-tuned on the
[BESSTIE-CW-26 dataset](https://huggingface.co/datasets/surrey-nlp/BESSTIE-CW-26).

Adapters are hosted under
[`{HF_USER}`](https://huggingface.co/{HF_USER}) on Hugging Face.
This application is a research prototype developed for the COMM061
Natural Language Processing project at the University of Surrey.
"""
        )

        predict_button.click(
            fn=predict,
            inputs=[text_input, variety_input],
            outputs=[
                prediction_output,
                confidence_output,
                details_output,
            ],
        )

    return demo


def main() -> None:
    """Load the models and launch the Gradio application."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    load_resources()
    demo = build_demo()

    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
        show_error=True,
    )


if __name__ == "__main__":
    main()
