import os
import torch
import pandas as pd
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

DATA_PATH = "./data"
MODEL_PATH = "./saved_model"
BASE_MODEL = "Salesforce/blip-image-captioning-base"

test_df = pd.read_csv(f"{DATA_PATH}/test.csv")
test_samples = test_df.sample(5, random_state=42)

print("\nLoading base model...")
base_processor = BlipProcessor.from_pretrained(BASE_MODEL)
base_model = BlipForConditionalGeneration.from_pretrained(BASE_MODEL).to(device)

print("\nLoading fine-tuned model...")
ft_processor = BlipProcessor.from_pretrained(MODEL_PATH)
ft_model = BlipForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)

for idx, row in test_samples.iterrows():
    image_path = os.path.join(DATA_PATH, row["image_path"])
    image = Image.open(image_path).convert("RGB")

    print("\n===================================================")
    print(f"IMAGE: {row['image_path']}")
    print("GROUND TRUTH:", row["caption"])

    # Base model with beam search
    inputs = base_processor(images=image, return_tensors="pt").to(device)
    base_output = base_model.generate(
        **inputs,
        max_length=50,
        num_beams=5,
        early_stopping=True
    )
    base_caption = base_processor.decode(base_output[0], skip_special_tokens=True)

    # Fine-tuned model with beam search
    inputs = ft_processor(images=image, return_tensors="pt").to(device)
    ft_output = ft_model.generate(
        **inputs,
        max_length=50,
        num_beams=5,
        early_stopping=True
    )
    ft_caption = ft_processor.decode(ft_output[0], skip_special_tokens=True)

    print("\nBase Model Caption (Beam):")
    print(base_caption)

    print("\nFine-Tuned Model Caption (Beam):")
    print(ft_caption)

print("\nEvaluation complete.")


