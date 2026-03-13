📰 Image-Driven News Article Generation
Using Fake Image Detection, Transformer-Based Captioning, and LLM Synthesis
📌 Overview

This project presents a complete multimodal AI pipeline for automatic news article generation from images.

The system consists of:

🔴 Transformer-based image caption generation (Core Research Contribution)

🛡 Fake image detection (Real vs AI-generated)

✍️ LLM-based structured news synthesis

🌐 End-to-end deployment using Streamlit

The primary scientific contribution lies in domain-adaptive fine-tuning of a Vision-Language Transformer (BLIP) for news caption generation.

All other modules are system-level extensions built around the captioning model.

🎯 Research Objective

To fine-tune a transformer-based multimodal captioning model on news-domain imagery and evaluate its improvement over a base pretrained model using standard captioning metrics.

🏗 Complete System Architecture
1️⃣ Dataset Collection & Training : 

┌──────────────────────┐
│   NYT News URLs      │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  Image URL Extraction│
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│    Image Download    │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  Caption Extraction  │
└──────────┬───────────┘
           ↓
┌──────────────────────────────┐
│ Dataset Cleaning & Normalize │
└──────────┬───────────────────┘
           ↓
┌──────────────────────┐
│ Structured Dataset   │
│        (CSV)         │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│    BLIP Base Model   │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│   Fine-Tuning Stage  │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Fine-Tuned BLIP Model│
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Evaluation: BLEU &   │
│        CIDEr         │
└──────────────────────┘

This stage represents the core research contribution.

2️⃣ Transformer Captioning Architecture (Model-Level)

┌──────────────────┐
│   Input Image    │
└────────┬─────────┘
         ↓
┌────────────────────────────┐
│ Vision Transformer Encoder │
│  (Patch Embedding + MHSA)  │
└────────┬───────────────────┘
         ↓
┌────────────────────────────┐
│     Image Embeddings       │
│  (Visual Feature Tokens)   │
└────────┬───────────────────┘
         ↓
┌────────────────────────────┐
│  Transformer Text Decoder  │
│ (Masked Self-Attention +   │
│  Cross Attention to Image) │
└────────┬───────────────────┘
         ↓
┌──────────────────┐
│ Generated Caption│
└──────────────────┘

Mathematical Formulation (GitHub-safe)

Vision Encoding:

I → Ev(I)

Autoregressive Caption Generation:

P(wt | w<t, V)

Training Objective:

L = - Σ log P(wt | w<t, V)

Where:

I = input image

V = image embeddings

wt = token at time t

L = cross-entropy loss

3️⃣ End-to-End Deployment Architecture

┌─────────────────────────┐
│   User Upload Image     │
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│   Fake Image Detector   │
│ (Deepfake / Manipulation│
│        Model)           │
└───────┬───────────┬─────┘
        │           │
      FAKE         REAL
        │           │
        ↓           ↓
┌──────────────┐   ┌────────────────────────────┐
│ Stop Process │   │ Fine-Tuned BLIP Caption    │
└──────────────┘   │        Generator           │
                   └────────────┬───────────────┘
                                ↓
                   ┌────────────────────────────┐
                   │   Prompt Template Builder  │
                   │ (Context + Instructions)   │
                   └────────────┬───────────────┘
                                ↓
                   ┌────────────────────────────┐
                   │   LLM Article Generator    │
                   └────────────┬───────────────┘
                                ↓
                   ┌────────────────────────────┐
                   │ Structured News Article    │
                   │          Output            │
                   └────────────────────────────┘



📂 Dataset Preparation

🔹 Data Sources

NYT News Article URLs

Scraped from official news pages

Extract associated images and captions

Public News Event Imagery

Open-source journalistic datasets

Event-based real-world images with annotations

🔹 Data Processing Steps

Extract Image URLs

Parse HTML content

Identify <img> tags and metadata

Download Images

Use requests or aiohttp

Validate image integrity

Store in /images directory

Extract Ground-Truth Captions

Capture original article captions

Map each caption to corresponding image

Normalize Captions

Convert to lowercase

Remove special characters

Remove extra whitespace

Standardize punctuation

Construct Structured CSV Dataset

Map image path ↔ cleaned caption

Ensure no duplicates

Remove corrupt samples

🔹 Dataset Format
image_path	caption
images/001.jpg	protesters gather outside parliament
images/002.jpg	a worker operates machinery in shanghai

🧠 Core Model: Transformer-Based Captioning

Base Model Used:

Salesforce/blip-image-captioning-base

Architecture Components
Vision Encoder

Uses Vision Transformer (ViT) to convert images into patch embeddings.

Transformer Decoder

Generates caption autoregressively conditioned on image embeddings.

🔧 Fine-Tuning Strategy

Parameter	Configuration
Dataset Size	~35000 samples
Optimizer	AdamW
Device	Apple Silicon (MPS)
Loss Function	Cross-Entropy
Training Strategy	Teacher Forcing

Model artifacts saved to:

saved_model/

📊 Evaluation Methodology

Metrics Used
BLEU

Measures n-gram overlap precision between generated and reference captions.

CIDEr

TF-IDF weighted consensus similarity metric optimized for image captioning.

Experimental Results

Metric	Base Model	Fine-Tuned	Improvement
BLEU	0.0033	0.0213	+537%
CIDEr	0.0426	0.0945	+121%
Interpretation

Significant improvement in domain adaptation

Better semantic alignment with news captions

Higher consensus similarity

🛡 Fake Image Detection (Extension)

Model Used:

umm-maybe/AI-image-detector

Type:

Vision Transformer classifier trained on real vs AI-generated images.

Decision Rule:

If AI probability ≥ 0.85 → Block article generation
Else → Continue

✍️ LLM-Based News Article Generation (Extension)

LLM Used:

llama-3.1-8b-instant (Groq API)

Process

Caption generated from BLIP

Prompt template constructed

News category inferred

Structured journalistic article generated

🌐 Deployment

Framework: Streamlit

Run locally:

conda activate newsenv
cd single_image_news_pipeline
python -m streamlit run web_app.py

📁 Repository Structure

news_caption_project/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── images/
│
├── saved_model/
│
├── train.py
├── caption_metrics.py
│
└── single_image_news_pipeline/
    ├── web_app.py
    ├── models/
    │   ├── blip_caption.py
    │   ├── fake_detector.py
    │   └── news_generator.py
    └── utils/
        └── news_templates.py
        
🧰 Technologies Used

PyTorch — Model training

Transformers — BLIP and AI detector

Evaluate — BLEU

pycocoevalcap — CIDEr

Streamlit — Web interface

Requests — LLM API

Pillow — Image processing

NumPy — Numerical operations

🔬 Key Contributions

Domain-adaptive transformer-based captioning

Quantitative evaluation with standard metrics

Integrated authenticity filtering

LLM-based structured article synthesis

Fully deployable multimodal pipeline

⚠️ Limitations

AI image detection is probabilistic

BLEU does not capture full semantic richness

LLM may introduce contextual hallucinations

Performance depends on dataset scale

🚀 Future Work

Larger-scale fine-tuning

Multi-reference evaluation

Human evaluation study

Cloud deployment

Model compression

👤 Author

Phani Inturi
AI Research – Multimodal Systems