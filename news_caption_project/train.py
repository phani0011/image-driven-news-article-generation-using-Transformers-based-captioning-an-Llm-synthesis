import torch
import os
import gc
import argparse
import pandas as pd
from datasets import Dataset
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    Trainer,
    TrainingArguments
)
from preprocess import preprocess

parser = argparse.ArgumentParser()
parser.add_argument("--chunk", type=int, required=True)
args = parser.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

DATA_PATH = "./data"
MODEL_PATH = "./saved_model"

TOTAL_TRAIN_SAMPLES = 30000   # 🔥 best stable subset
CHUNK_SIZE = 1500
EPOCHS = 1
LR = 5e-5

train_df_full = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))

# Random stable sampling (no bias)
train_df_full = train_df_full.sample(
    TOTAL_TRAIN_SAMPLES,
    random_state=42
).reset_index(drop=True)

total_samples = len(train_df_full)
num_chunks = total_samples // CHUNK_SIZE

print(f"Total training samples used: {total_samples}")
print(f"Total chunks: {num_chunks}")

if chunk_idx >= num_chunks:
    print("Chunk index exceeds available chunks.")
    exit()

start = chunk_idx * CHUNK_SIZE
end = start + CHUNK_SIZE

print(f"\n==== Training chunk {chunk_idx+1}/{num_chunks} ====")
print(f"Samples {start} to {end}")

train_df = train_df_full[start:end]
train_dataset = Dataset.from_pandas(train_df)

# ================= LOAD MODEL =================
model_id = "Salesforce/blip-image-captioning-base"

if os.path.exists(MODEL_PATH):
    print("Loading previously saved model...")
    model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH)
    processor = BlipProcessor.from_pretrained(MODEL_PATH)
else:
    print("Loading base model...")
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    processor = BlipProcessor.from_pretrained(model_id)

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 2 decoder layers + LM head
    for name, param in model.named_parameters():
        if "text_decoder.bert.encoder.layer.10" in name:
            param.requires_grad = True
        if "text_decoder.bert.encoder.layer.11" in name:
            param.requires_grad = True
        if "lm_head" in name:
            param.requires_grad = True

model.to(device)

# Print stats only first time
if chunk_idx == 0:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable}")
    print(f"Total params: {total}")
    print(f"Trainable %: {100 * trainable / total:.2f}%")

# ================= PREPROCESS =================
train_dataset = train_dataset.map(
    lambda x: preprocess(x, processor, DATA_PATH),
    remove_columns=train_dataset.column_names,
    load_from_cache_file=False
)

# ================= TRAIN =================
training_args = TrainingArguments(
    output_dir="./tmp",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=50,
    save_strategy="no",
    report_to="none",
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

# ================= SAVE MODEL =================
print("Saving updated model...")
model.save_pretrained(MODEL_PATH)
processor.save_pretrained(MODEL_PATH)

# ================= CLEAN MEMORY =================
del train_dataset
gc.collect()
torch.mps.empty_cache()

print("Chunk completed successfully.")

