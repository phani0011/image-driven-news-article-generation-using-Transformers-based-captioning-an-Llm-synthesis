import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


class FakeImageDetector:

    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.processor = AutoImageProcessor.from_pretrained(
            "umm-maybe/AI-image-detector"
        )

        self.model = AutoModelForImageClassification.from_pretrained(
            "umm-maybe/AI-image-detector"
        ).to(self.device)

        self.threshold = 0.85   # High confidence threshold

    def check_real_or_fake(self, image_input):

        image = Image.open(image_input).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()

        label_map = self.model.config.id2label
        predicted_label = label_map[pred_class].lower()

        # Apply strict blocking only if strong confidence AI
        if predicted_label == "ai" and confidence >= self.threshold:
            return "FAKE", round(confidence, 3)

        return "REAL", round(1 - confidence, 3)