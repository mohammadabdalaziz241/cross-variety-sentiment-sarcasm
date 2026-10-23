"""Hugging Face Space: sarcasm detection across British, Australian and Indian English.

One shared Gemma-2-2B base model with three variety-specific LoRA adapters.
On a GPU the base is loaded in 4-bit NF4 (as in training); on the free CPU tier
it is loaded in bfloat16, so probabilities can differ slightly from the
reported 4-bit results.
"""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LOGGER = logging.getLogger("sarcasm-space")

BASE_MODEL = os.getenv("BASE_MODEL", "google/gemma-2-2b")
HF_USER = os.getenv("HF_USER", "Mohammadeeu20")
HF_TOKEN = os.getenv("HF_TOKEN")  # set as a Space secret; Gemma is gated
MAX_LENGTH = 384

ADAPTERS = {
    "British (en-UK)": ("en-UK", f"{HF_USER}/besstie-sarcasm-gemma2b-lora-en-uk"),
    "Australian (en-AU)": ("en-AU", f"{HF_USER}/besstie-sarcasm-gemma2b-lora-en-au"),
    "Indian (en-IN)": ("en-IN", f"{HF_USER}/besstie-sarcasm-gemma2b-lora-en-in"),
}

EXAMPLES = [
    ["Oh fantastic, another rail strike. Just what I needed this Monday morning.", "British (en-UK)"],
    ["Yeah nah, the LNP really nailed it with this budget. Brilliant stuff.", "Australian (en-AU)"],
    ["Wah, the new traffic rules are working so beautifully. Absolutely no jams now.", "Indian (en-IN)"],
    ["I genuinely enjoyed the film, the cinematography was stunning.", "British (en-UK)"],
    ["The match yesterday was a cracker, both teams played really well.", "Australian (en-AU)"],
]


def _load_base():
    if torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        return AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=2, quantization_config=quant,
            torch_dtype=torch.float16, device_map="auto", token=HF_TOKEN,
        ), "GPU, 4-bit NF4"
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    return AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, token=HF_TOKEN,
    ), "CPU, bfloat16"


@lru_cache(maxsize=1)
def load():
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base, mode = _load_base()
    base.config.pad_token_id = tokenizer.pad_token_id

    model = None
    for name, repo in ADAPTERS.values():
        if model is None:
            model = PeftModel.from_pretrained(base, repo, adapter_name=name, token=HF_TOKEN)
        else:
            model.load_adapter(repo, adapter_name=name, token=HF_TOKEN)
    model.eval()
    LOGGER.info("Loaded base + 3 adapters (%s) in %.1f s", mode, time.perf_counter() - start)
    return tokenizer, model, mode


def predict(text: str, variety: str):
    text = (text or "").strip()
    if not text:
        return {}, "Please enter some text."
    if variety not in ADAPTERS:
        return {}, f"Unknown variety: {variety}"

    tokenizer, model, mode = load()
    adapter_name, _ = ADAPTERS[variety]
    model.set_adapter(adapter_name)

    start = time.perf_counter()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = inputs.to(model.get_input_embeddings().weight.device)
    with torch.inference_mode():
        probs = model(**inputs).logits.float().softmax(-1)[0].cpu().tolist()
    latency_ms = (time.perf_counter() - start) * 1000

    label = {"Sarcastic": probs[1], "Not sarcastic": probs[0]}
    details = f"Adapter `{adapter_name}` · {mode} · {latency_ms:.0f} ms"
    return label, details


with gr.Blocks(title="Sarcasm Detector: British, Australian & Indian English") as demo:
    gr.Markdown(
        """
# Sarcasm detector for British, Australian and Indian English

Gemma-2-2B with three variety-specific QLoRA adapters (about 25 MB each) that share one base model
and are swapped per request. Pick the English variety closest to the text.

[Model cards](https://huggingface.co/Mohammadeeu20) ·
[Code and results](https://github.com/mohammadabdalaziz241/cross-variety-sentiment-sarcasm)
"""
    )
    with gr.Row():
        with gr.Column(scale=2):
            text_in = gr.Textbox(label="Text", lines=4, placeholder="Type or paste a sentence...")
            variety_in = gr.Dropdown(list(ADAPTERS), value="British (en-UK)", label="English variety")
            button = gr.Button("Detect sarcasm", variant="primary")
        with gr.Column(scale=1):
            label_out = gr.Label(label="Prediction", num_top_classes=2)
            details_out = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=[text_in, variety_in])
    gr.Markdown(
        "Research prototype from a University of Surrey NLP project. The Indian English adapter is "
        "the weakest (sarcastic-class F1 about 0.12), so treat its sarcasm predictions with caution. "
        "On the free CPU tier the first request loads the model and takes a minute or two."
    )
    button.click(predict, inputs=[text_in, variety_in], outputs=[label_out, details_out])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    demo.queue(max_size=8).launch()
