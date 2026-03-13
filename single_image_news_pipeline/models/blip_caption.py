import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image


class BLIPCaptionGenerator:

    def __init__(self, model_path):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[INFO] Using device: {self.device}")

        self.processor = BlipProcessor.from_pretrained(model_path)
        self.model = BlipForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def generate_caption(self, image_path):

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=40,
                num_beams=3,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )

        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption
