# Cross-Variety Sentiment and Sarcasm Classification

An end-to-end natural language processing study of sentiment and sarcasm detection across British, Australian, and Indian English using classical machine learning, transformer fine-tuning, and parameter-efficient adaptation.

This project investigates how linguistic variety, domain composition, code-mixing, and class imbalance affect model performance and cross-variety generalisation.

## Project Overview

The project uses the BESSTIE-CW-26 dataset, containing text from three English varieties:

* British English (`en-UK`)
* Australian English (`en-AU`)
* Indian English (`en-IN`)

Two binary classification tasks are studied:

* Sentiment classification
* Sarcasm detection

The complete pipeline includes:

* Dataset exploration and visualisation
* Vocabulary and linguistic-distance analysis
* Hindi-English code-mixing analysis
* TF-IDF classical machine-learning baselines
* RoBERTa fine-tuning
* Gemma-2-2B adaptation using LoRA
* Cross-variety transfer evaluation
* Error analysis
* Zero-shot and few-shot prompting
* Gradio-based deployment
* Accuracy, latency, and model-size benchmarking

## Key Results

| Experiment                             | Macro-F1 |
| -------------------------------------- | -------: |
| RoBERTa — pooled sentiment             |    0.902 |
| RoBERTa — pooled sarcasm               |    0.709 |
| Gemma-2-2B + LoRA — best en-AU adapter |    0.793 |
| TF-IDF + Logistic Regression — sarcasm |    0.621 |

Additional findings:

* Cross-variety sarcastic-class F1 dropped by approximately 37% compared with in-variety evaluation.
* British and Australian English transferred more effectively between one another than either transferred to Indian English.
* Indian English contained substantial Hindi-English code-mixing, contributing to a larger linguistic and transfer gap.
* LoRA adapters trained only approximately 0.4% of Gemma-2-2B parameters.
* Each trained adapter required approximately 25 MB of storage.
* RoBERTa provided the strongest overall balance between predictive performance and inference efficiency.

## Methodology

### Dataset Analysis

The exploratory analysis examined:

* Class distributions
* Source-domain distributions
* Text lengths
* Vocabulary overlap
* TF-IDF cosine similarity
* Jaccard similarity
* Variety-specific vocabulary
* Hindi-English code-mixing

### Classical Baselines

The classical experiments used:

* TF-IDF features
* Logistic Regression
* Linear Support Vector Machines
* Class-weighted training

### Transformer Fine-Tuning

RoBERTa-base was fine-tuned for sentiment and sarcasm classification using:

* Weighted cross-entropy loss
* Validation Macro-F1 for checkpoint selection
* Multiple random seeds
* Pooled and variety-specific training

### Parameter-Efficient Adaptation

Gemma-2-2B was adapted using:

* 4-bit NF4 quantisation
* Low-Rank Adaptation
* Paged AdamW
* Variety-specific adapters
* Cross-variety testing

### Evaluation

The primary metric was Macro-F1 because sarcasm was strongly imbalanced.

Additional evaluation included:

* Accuracy
* Macro precision
* Macro recall
* Per-class F1
* Confusion matrices
* Cross-variety transfer matrices
* Seed stability
* Inference latency
* Model and adapter storage

## Repository Structure

```text
cross-variety-sentiment-sarcasm/
├── README.md
├── .gitignore
├── requirements.txt
├── requirements-full.txt
├── figures/
├── notebooks/
│   └── complete_experiments.ipynb
├── results/
└── src/
    └── app.py
```

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/mohammadabdalaziz241/cross-variety-sentiment-sarcasm.git
cd cross-variety-sentiment-sarcasm

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

The exact original experimental environment is preserved in:

```text
requirements-full.txt
```

## Running the Experiments

Open the experiment notebook:

```bash
jupyter lab notebooks/complete_experiments.ipynb
```

Some experiments, particularly Gemma-2-2B quantisation and LoRA training, require a CUDA-compatible GPU with sufficient memory.

The notebook has been cleaned of stored outputs and machine-specific paths. Training sections may require substantial computation and access to Hugging Face models.

## Running the Gradio Application

Run:

```bash
python src/app.py
```

The application loads the shared Gemma-2-2B backbone and variety-specific LoRA adapters.

A compatible NVIDIA GPU is recommended.

## Figures

The `figures/` directory contains:

* Label-distribution analysis
* Sarcasm distribution by source
* Text-length distributions
* Vocabulary-similarity matrices
* Cross-variety transfer matrices
* RoBERTa and LoRA comparisons
* Confusion matrices
* Few-shot prompting examples
* Gradio application examples
* Accuracy-latency comparisons

## Results

The `results/` directory contains the aggregate experiment outputs used to generate the final comparisons, including:

* Classical baseline summaries
* RoBERTa results
* Cross-variety transfer matrices
* LoRA transfer matrices
* Final sentiment and sarcasm comparisons
* Error-analysis outputs
* Few-shot prompting results
* Efficiency measurements

## Contributors

This project was completed as part of the COMM061 Natural Language Processing module at the University of Surrey.

* Mohammad Abdalaziz
* Talha Rizwan
* Ananya Agarwal
* Zain Ul Abideen
* Yinan Lyu

### Mohammad Abdalaziz’s Contributions

Mohammad contributed across the complete project lifecycle, including:

* Dataset analysis and visualisation
* Linguistic-distance and code-mixing analysis
* Classical machine-learning baselines
* RoBERTa fine-tuning
* Gemma-2-2B and LoRA experiments
* Cross-variety transfer evaluation
* Error analysis
* Zero-shot and few-shot prompting
* Gradio deployment
* Efficiency benchmarking
* Result interpretation
* Figure generation
* Report writing and project integration

## Academic Context

This repository is a cleaned portfolio version of a university group project.

Student identification numbers, originality declarations, signatures, submission documents, and private assessment material are intentionally excluded.

## Limitations

* Sarcasm labels are highly imbalanced across varieties.
* Language-variety effects are partly confounded by source-domain differences.
* Several experiments require high-memory GPU hardware.
* The deployed interface is intended as a research prototype rather than a production service.
* The LoRA adapters may specialise strongly to their training variety and transfer poorly to other varieties.

## Acknowledgements

This work was completed at the University of Surrey as part of the COMM061 Natural Language Processing module.

