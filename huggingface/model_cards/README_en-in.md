---
base_model: google/gemma-2-2b
library_name: peft
license: gemma
language:
- en
tags:
- lora
- qlora
- peft
- gemma
- text-classification
- sarcasm-detection
- besstie
datasets:
- surrey-nlp/BESSTIE-CW-26
pipeline_tag: text-classification
---

# Sarcasm detection LoRA adapter for Indian English (en-IN)

A 4-bit QLoRA adapter that turns [`google/gemma-2-2b`](https://huggingface.co/google/gemma-2-2b) into a binary sarcasm classifier for **Indian English** text (Google reviews and Reddit posts).
It is one of three variety-specific adapters (British, Australian, Indian English) that share a single base model and can be swapped at request time.

- **Try it:** [live demo Space](https://huggingface.co/spaces/Mohammadeeu20/besstie-sarcasm-detector)
- **Code and full study:** [github.com/mohammadabdalaziz241/cross-variety-sentiment-sarcasm](https://github.com/mohammadabdalaziz241/cross-variety-sentiment-sarcasm)
- **Sibling adapters:** [en-UK](https://huggingface.co/Mohammadeeu20/besstie-sarcasm-gemma2b-lora-en-uk) · [en-AU](https://huggingface.co/Mohammadeeu20/besstie-sarcasm-gemma2b-lora-en-au) · [en-IN](https://huggingface.co/Mohammadeeu20/besstie-sarcasm-gemma2b-lora-en-in)

## Results

Test-set results for this adapter (seed 123, the uploaded checkpoint), evaluated on each English variety:

| Test variety | Macro-F1 | Sarcastic-class F1 | Accuracy |
|---|---:|---:|---:|
| en-UK (British English) | 0.552 | 0.140 | 0.930 |
| en-AU (Australian English) | 0.440 | 0.050 | 0.712 |
| en-IN (Indian English) **(in-variety)** | 0.542 | 0.121 | 0.929 |

Across two seeds, the in-variety Macro-F1 is **0.565 ± 0.032**, compared with 0.596 for full RoBERTa-base fine-tuning on the same split.

The weakest adapter. Indian English has only about 95 sarcastic training examples and heavy Hindi-English code-mixing, and the low-rank adapter struggles to learn the sarcastic class (F1 0.12). Treat its sarcasm predictions with caution.

**Efficiency:** about 0.4% of parameters trained (6.4M of 1.6B), and the adapter is about 25 MB on disk, against roughly 500 MB for a full RoBERTa checkpoint. Measured single-example GPU latency was about 39 ms with the 4-bit base.

## How to use

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
base = AutoModelForSequenceClassification.from_pretrained(
    "google/gemma-2-2b", num_labels=2, quantization_config=bnb,
    torch_dtype=torch.float16, device_map="auto")
base.config.pad_token_id = tokenizer.pad_token_id
model = PeftModel.from_pretrained(base, "Mohammadeeu20/besstie-sarcasm-gemma2b-lora-en-in").eval()

inputs = tokenizer("Oh brilliant, another delayed train.", return_tensors="pt").to(model.device)
with torch.inference_mode():
    p_sarcastic = model(**inputs).logits.float().softmax(-1)[0, 1].item()
print(f"P(sarcastic) = {p_sarcastic:.3f}")
```

Label 1 = sarcastic, label 0 = not sarcastic. Gemma is a gated model, so accept its licence on Hugging Face and log in before loading.

## Training details

- **Data:** [surrey-nlp/BESSTIE-CW-26](https://huggingface.co/datasets/surrey-nlp/BESSTIE-CW-26), en-IN training split only
- **Base model:** google/gemma-2-2b loaded in 4-bit NF4 with double quantisation (QLoRA)
- **LoRA:** r = 16, alpha = 32, dropout = 0.05, targets q_proj, k_proj, v_proj, o_proj; sequence-classification head
- **Optimisation:** 4 epochs, learning rate 2e-4, effective batch size 16, warm-up 10%, weight decay 0.01, paged AdamW 8-bit, class-weighted cross-entropy
- **Selection:** best epoch by validation Macro-F1; max sequence length 384 tokens

## Limitations

- Sarcasm is rare and subjective; the minority-class F1 is the number to watch, not accuracy.
- Adapters over-specialise to their own variety: cross-variety transfer is clearly weaker (see the table).
- Trained on short reviews and Reddit posts; other domains are untested.
- Research prototype, not intended for moderation or other high-stakes decisions.

## Context

Built as part of a team project for the COMM061 Natural Language Processing module at the University of Surrey (2026). Author of this release: Mohammad Abdalaziz.
