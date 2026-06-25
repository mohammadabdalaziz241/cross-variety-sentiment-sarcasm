#BESSTIE Sarcasm Detector (Gradio deployment)
#A user interface for sarcasm detection on british (en-UK), australian (en-AU),
#and indian (en-IN) english, based on Gemma-2-2B and three variety specific LoRA adapters from HuggingFace Hub.
#Usage: python app.py
#The app loads the base model once,and then registers all three adapters by name. Variety switches at request time are O(0.01s) the
#"A ship of tiny adapters,and a cheap switch" deployment of Lora.
#COMM061 NLP coursework, University of Surrey, 2026.

import time
import torch
import gradio as gr
from transformers import (AutoModelForSequenceClassification,AutoTokenizer,BitsAndBytesConfig,)
from peft import PeftModel
#configuration 
BASE_MODEL = "google/gemma-2-2b"
HF_USER    = "Mohammadeeu20"
ADAPTER_REPOS = {"British (en-UK)":   f"{HF_USER}/besstie-sarcasm-gemma2b-lora-en-uk","Australian (en-AU)": f"{HF_USER}/besstie-sarcasm-gemma2b-lora-en-au","Indian (en-IN)":    f"{HF_USER}/besstie-sarcasm-gemma2b-lora-en-in",}
ADAPTER_NAMES = {"British (en-UK)":   "en-UK","Australian (en-AU)": "en-AU","Indian (en-IN)": "en-IN",}
MAX_LENGTH = 384
print("BESSTIE Sarcasm Detector starting up")
print(f"Loading tokenizer ({BASE_MODEL})")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"[2/3] Loading base model in 4-bit (Gemma-2-2B)")
bnb_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16,)
t0 = time.time()
base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL,num_labels=2,quantization_config=bnb_config,torch_dtype=torch.float16,device_map="auto",)
base_model.config.pad_token_id = tokenizer.pad_token_id
print(f"base model loaded in {time.time()-t0:.1f}s")
print(f"[3/3] Attaching 3 LoRA adapters from HF Hub")
model = None
for label, repo_id in ADAPTER_REPOS.items():
    name = ADAPTER_NAMES[label]
    t0 = time.time()
    if model is None:
        model = PeftModel.from_pretrained(base_model, repo_id, adapter_name=name)
    else:
        model.load_adapter(repo_id, adapter_name=name)
    print(f"{label:<22} {repo_id}({time.time()-t0:.1f}s)")
model.eval()
print(f"Ready. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")
print(f"Registered adapters: {list(model.peft_config.keys())}")

# the prediction function
def predict(text, variety_label):
    if not text or not text.strip():
        return "—", 0.0, "Please enter some text."
    if variety_label not in ADAPTER_NAMES:
        return "—", 0.0, f"Unknown variety: {variety_label}"
    adapter_name = ADAPTER_NAMES[variety_label]
    model.set_adapter(adapter_name)
    t0 = time.time()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH,).to(model.device)
    with torch.no_grad():
        out = model(**inputs)
    probs = torch.softmax(out.logits.float(), dim=-1).cpu().numpy()[0]
    p_not_sarc, p_sarc = float(probs[0]), float(probs[1])
    pred = int(p_sarc >= 0.5)
    latency_ms = (time.time() - t0) * 1000
    label = "Sarcastic" if pred == 1 else "Not sarcastic"
    confidence = p_sarc if pred == 1 else p_not_sarc
    info = (f"Variety: **{variety_label}** (adapter: `{adapter_name}`) · "f"P(sarcastic) = {p_sarc:.3f} · "f"P(not sarcastic) = {p_not_sarc:.3f} · "f"latency: {latency_ms:.0f} ms")
    return label, confidence, info
# example texts
EXAMPLES = [
    ["Oh fantastic, another rail strike. Just what I needed this Monday morning.", "British (en-UK)"],
    ["Yeah nah, the LNP really nailed it with this budget. Brilliant stuff.", "Australian (en-AU)"],
    ["Wah, the new traffic rules are working so beautifully. Absolutely no jams now.", "Indian (en-IN)"],
    ["I genuinely enjoyed the film — the cinematography was stunning.", "British (en-UK)"],
    ["The match yesterday was a cracker, both teams played really well.", "Australian (en-AU)"],
    ["The food at the new restaurant is quite good, would recommend.", "Indian (en-IN)"],
]
# gradio ui
with gr.Blocks(title="BESSTIE Sarcasm Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        #BESSTIE Sarcasm Detector
        Detect sarcasm in **British, Australian, and Indian English** using
        variety specific LoRA adapters on Gemma-2-2B.
        Select the English variety closest to the text's origin — the
        adapter swap happens in milliseconds because we ship tiny LoRA
        deltas (≈25 MB each) on top of one shared base model.
        """
    )
    with gr.Row():
        with gr.Column(scale=2):
            text_in = gr.Textbox(label="Text to classify",placeholder="Type or paste a sentence…",lines=4,)
            variety_in = gr.Dropdown(choices=list(ADAPTER_NAMES.keys()),value="British (en-UK)",label="English variety",)
            predict_btn = gr.Button("Detect sarcasm", variant="primary")
        with gr.Column(scale=1):
            label_out = gr.Label(label="Prediction")
            confidence_out = gr.Slider(label="Confidence", minimum=0, maximum=1, value=0,interactive=False,)
            info_out = gr.Markdown(label="Details")
    gr.Examples(examples=EXAMPLES,inputs=[text_in, variety_in],label="Try these examples",)
    gr.Markdown(
        """
        **About** The model is Gemma-2-2B (4-bit quantized) with one of three
        LoRA adapters fine-tuned on the [BESSTIE](https://huggingface.co/datasets/surrey-nlp/BESSTIE-CW-26)
        sarcasm dataset. Adapters are hosted on HuggingFace Hub
        (`Mohammadeeu20/besstie-sarcasm-gemma2b-lora-*`). Built for the
        COMM061 NLP coursework at the University of Surrey, 2026.
        """
    )
    predict_btn.click(fn=predict,inputs=[text_in, variety_in],outputs=[label_out, confidence_out, info_out],)
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1",server_port=7860,share=False,show_error=True,)
