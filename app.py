"""Streamlit demo: upload a chest X-ray, get a probability and a Grad-CAM overlay.

    streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from xray.data import build_transforms
from xray.gradcam import GradCAM, overlay
from xray.model import build_model, gradcam_layer

CKPT = Path("checkpoints")

st.set_page_config(page_title="Chest X-ray pneumonia screening", layout="wide")
st.title("Chest X-ray pneumonia screening (research demo)")
st.warning("Research prototype only. Not a medical device and not for diagnosis.")


@st.cache_resource
def load():
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(CKPT / "best.pt", map_location="cpu"))
    threshold = json.loads((CKPT / "threshold.json").read_text())["threshold"]
    return model.eval(), threshold


uploaded = st.file_uploader("Upload a frontal chest X-ray (PNG/JPEG)", type=["png", "jpg", "jpeg"])
if uploaded:
    model, threshold = load()
    image = Image.open(uploaded).convert("RGB")
    _, eval_tf = build_transforms()
    x = eval_tf(image).unsqueeze(0)

    cam = GradCAM(model, gradcam_layer(model))
    heatmap, logits = cam(x)
    cam.remove()
    prob = torch.sigmoid(logits)[0, 0].item()

    left, right = st.columns(2)
    left.image(image, caption="Input", use_container_width=True)
    right.image(overlay(np.array(image.resize((224, 224))), heatmap), caption="Grad-CAM", use_container_width=True)

    label = "PNEUMONIA suspected" if prob >= threshold else "No pneumonia pattern detected"
    st.metric(label, f"{prob:.1%}", help=f"Decision threshold {threshold:.3f}, tuned for high sensitivity")
