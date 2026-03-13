import streamlit as st
from PIL import Image
import torch

from models.blip_caption import BLIPCaptionGenerator
from models.fake_detector import FakeImageDetector
from models.news_generator import NewsGenerator
from utils.news_templates import generate_prompt


# ================= CONFIG =================
BLIP_MODEL_PATH = "../saved_model"   # because we are inside single_image_news_pipeline


# ================= PAGE SETUP =================
st.set_page_config(page_title="AI Verified News Generator", layout="wide")

st.title("📰 AI Verified Automatic News Generator")
st.write("Upload an image. If verified as REAL, a news article will be generated automatically.")


# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)


# ================= MAIN PIPELINE =================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ---------- Step 1: Authenticity Check ----------
    st.write("### 🔍 Checking Authenticity...")

    fake_detector = FakeImageDetector()
    authenticity, confidence = fake_detector.check_real_or_fake(uploaded_file)

    st.write(f"Detection Result: **{authenticity}**")
    st.write(f"Confidence: {confidence}")

    if authenticity.upper() != "REAL":
        st.error("❌ AI-Generated / Fake Image Detected. News generation blocked.")
        st.stop()

    st.success("✅ Image Verified as REAL")


    # ---------- Step 2: Caption Generation ----------
    if st.button("Generate News Article"):

        st.write("### ✍ Generating Caption...")

        caption_model = BLIPCaptionGenerator(BLIP_MODEL_PATH)
        caption = caption_model.generate_caption(uploaded_file)

        st.write(f"Generated Caption: {caption}")


        # ---------- Step 3: LLM Prompt Construction ----------
        st.write("### 🧠 Constructing News Prompt...")

        prompt = generate_prompt(caption, authenticity)


        # ---------- Step 4: LLM News Generation ----------
        st.write("### 📰 Generating News Article...")

        news_generator = NewsGenerator()
        article = news_generator.generate_news(prompt)

        st.markdown("## 📰 Generated News Article")
        st.write(article)

        st.success("✅ News Article Generated Successfully")
