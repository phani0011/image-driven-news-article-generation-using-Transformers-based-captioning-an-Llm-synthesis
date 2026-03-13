import os
from PIL import Image

def preprocess(example, processor, data_path, max_length=32):
    image_path = os.path.join(data_path, example["image_path"])
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        text=example["caption"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    image.close()

    inputs = {k: v.squeeze() for k, v in inputs.items()}
    inputs["labels"] = inputs["input_ids"].clone()

    return inputs
