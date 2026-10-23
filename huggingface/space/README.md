---
title: Sarcasm Detector (British, Australian, Indian English)
emoji: 🎭
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
license: gemma
short_description: Gemma-2-2B + variety-specific QLoRA adapters for sarcasm
models:
- google/gemma-2-2b
- Mohammadeeu20/besstie-sarcasm-gemma2b-lora-en-uk
- Mohammadeeu20/besstie-sarcasm-gemma2b-lora-en-au
- Mohammadeeu20/besstie-sarcasm-gemma2b-lora-en-in
---

# Sarcasm detector for British, Australian and Indian English

One shared Gemma-2-2B base model with three LoRA adapters, swapped per request.
Code and full results: https://github.com/mohammadabdalaziz241/cross-variety-sentiment-sarcasm

**Setup note:** Gemma is a gated model, so this Space needs a `HF_TOKEN` secret
(Settings → Variables and secrets) from an account that has accepted the Gemma licence.
