import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import evaluate
import matplotlib.pyplot as plt
import numpy as np

from transformers import BlipProcessor, BlipForConditionalGeneration
from pycocoevalcap.cider.cider import Cider
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ================= DEVICE =================
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)


# ================= PARAMETERS =================
TARGET_TEST_SAMPLES = 2000
BATCH_SIZE = 16
DATA_PATH = "./data"
FINE_TUNED_MODEL_PATH = "./saved_model"
BASE_MODEL_NAME = "Salesforce/blip-image-captioning-base"


# ================= LOAD BLEU =================
bleu_metric = evaluate.load("bleu")


# ================= LOAD TEST DATA =================
df_test = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
df_test = df_test.sample(TARGET_TEST_SAMPLES, random_state=42).reset_index(drop=True)

print(f"Evaluating on {len(df_test)} samples")


# ================= GENERATION SETTINGS =================
gen_args = {
    "max_length": 60,
    "min_length": 10,
    "num_beams": 5,
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 3,
    "length_penalty": 1.0,
    "early_stopping": True
}


# =========================================================
# MODEL EVALUATION
# =========================================================
def evaluate_model(model, processor):

    model.eval()

    predictions = []
    references = []

    with torch.no_grad():

        for i in tqdm(range(0, len(df_test), BATCH_SIZE)):

            batch_df = df_test.iloc[i:i+BATCH_SIZE]

            images = []
            captions = []

            for _, row in batch_df.iterrows():

                img_path = os.path.join(DATA_PATH, row["image_path"])
                image = Image.open(img_path).convert("RGB")

                images.append(image)
                captions.append(row["caption"].strip().lower())

            inputs = processor(
                images=images,
                return_tensors="pt",
                padding=True
            ).to(device)

            outputs = model.generate(**inputs, **gen_args)
            decoded = processor.batch_decode(outputs, skip_special_tokens=True)

            for pred, ref in zip(decoded, captions):

                predictions.append(pred.strip().lower())
                references.append([ref])

            del inputs, outputs

    # BLEU
    bleu_score = bleu_metric.compute(
        predictions=predictions,
        references=references
    )["bleu"]

    # CIDEr
    cider_scorer = Cider()

    gts = {i: [references[i][0]] for i in range(len(references))}
    res = {i: [predictions[i]] for i in range(len(predictions))}

    cider_score, _ = cider_scorer.compute_score(gts, res)

    return bleu_score, cider_score, predictions, references

# =========================================================
# GRAPH FUNCTIONS
# =========================================================
def plot_comparison_graphs(results):

    df = pd.DataFrame(results)

    models = df["model"]
    bleu = df["bleu"]
    cider = df["cider"]

    plt.figure(figsize=(6,4))
    plt.bar(models, bleu)
    plt.title("BLEU Score Comparison")
    plt.ylabel("BLEU Score")
    plt.savefig("bleu_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.bar(models, cider)
    plt.title("CIDEr Score Comparison")
    plt.ylabel("CIDEr Score")
    plt.savefig("cider_comparison.png", dpi=300)
    plt.close()

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(7,4))
    plt.bar(x - width/2, bleu, width, label="BLEU")
    plt.bar(x + width/2, cider, width, label="CIDEr")

    plt.xticks(x, models)
    plt.legend()
    plt.title("Model Performance Comparison")

    plt.savefig("model_comparison.png", dpi=300)
    plt.close()


# =========================================================
# SENTENCE LENGTH ANALYSIS
# =========================================================
def sentence_length_analysis(preds, refs):

    smooth = SmoothingFunction().method1

    lengths = []
    scores = []

    for pred, ref in zip(preds, refs):

        ref_text = ref[0]
        length = len(ref_text.split())

        score = sentence_bleu(
            [ref_text.split()],
            pred.split(),
            smoothing_function=smooth
        )

        lengths.append(length)
        scores.append(score)

    return lengths, scores


def plot_sentence_length_performance(base_preds, base_refs, ft_preds, ft_refs):

    base_len, base_bleu = sentence_length_analysis(base_preds, base_refs)
    ft_len, ft_bleu = sentence_length_analysis(ft_preds, ft_refs)

    bins = [0,5,10,15,20,25,30]

    base_avg = []
    ft_avg = []

    for i in range(len(bins)-1):

        b_scores = [s for l,s in zip(base_len, base_bleu) if bins[i] <= l < bins[i+1]]
        f_scores = [s for l,s in zip(ft_len, ft_bleu) if bins[i] <= l < bins[i+1]]

        base_avg.append(np.mean(b_scores) if b_scores else 0)
        ft_avg.append(np.mean(f_scores) if f_scores else 0)

    labels = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]

    plt.figure(figsize=(8,5))

    plt.plot(labels, base_avg, marker='o', label="Base BLIP")
    plt.plot(labels, ft_avg, marker='o', label="Fine-Tuned BLIP")

    plt.xlabel("Sentence Length (words)")
    plt.ylabel("Average BLEU Score")

    plt.title("Sentence Length vs Model Performance")

    plt.legend()
    plt.grid(True)

    plt.savefig("sentence_length_vs_performance.png", dpi=300)
    plt.close()


# =========================================================
# LOAD MODELS
# =========================================================
print("\nLoading Base Model...")
processor_base = BlipProcessor.from_pretrained(BASE_MODEL_NAME)
model_base = BlipForConditionalGeneration.from_pretrained(BASE_MODEL_NAME).to(device)

print("\nLoading Fine-Tuned Model...")
processor_ft = BlipProcessor.from_pretrained(FINE_TUNED_MODEL_PATH)
model_ft = BlipForConditionalGeneration.from_pretrained(FINE_TUNED_MODEL_PATH).to(device)


# =========================================================
# EVALUATE MODELS
# =========================================================
print("\nEvaluating Base Model...")
base_bleu, base_cider, base_preds, base_refs = evaluate_model(model_base, processor_base)

print("\nEvaluating Fine-Tuned Model...")
ft_bleu, ft_cider, ft_preds, ft_refs = evaluate_model(model_ft, processor_ft)


# =========================================================
# CLASSIFICATION METRICS
# =========================================================
base_acc, base_prec, base_rec, base_f1 = compute_classification_metrics(base_preds, base_refs)
ft_acc, ft_prec, ft_rec, ft_f1 = compute_classification_metrics(ft_preds, ft_refs)


# =========================================================
# PRINT RESULTS
# =========================================================
print("\n========== FINAL METRIC COMPARISON ==========\n")

def improvement(base, ft):
    return ((ft - base) / base * 100) if base != 0 else 0

print(f"BLEU  : {base_bleu:.6f} -> {ft_bleu:.6f}")
print(f"CIDEr : {base_cider:.6f} -> {ft_cider:.6f}")


# =========================================================
# GRAPH RESULTS
# =========================================================
results = [
    {"model": "Base BLIP", "bleu": base_bleu, "cider": base_cider},
    {"model": "Fine-Tuned BLIP", "bleu": ft_bleu, "cider": ft_cider}
]

plot_comparison_graphs(results)

plot_sentence_length_performance(
    base_preds,
    base_refs,
    ft_preds,
    ft_refs
)
