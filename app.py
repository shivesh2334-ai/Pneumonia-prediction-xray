

import os
import torch
import streamlit as st
from monai.networks.nets import DenseNet
from monai.transforms import Compose, ScaleIntensity, EnsureChannelFirst
# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = DenseNet(spatial_dims=2, in_channels=1, out_channels=2).to(device)

# Flexible model path
MODEL_PATH = os.getenv("MODEL_PATH", "best_metric_model.pth")

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    st.warning(
        f"⚠️ Model file not found at '{MODEL_PATH}'. "
        "Please upload your trained model file (.pth)."
    )
    uploaded_file = st.file_uploader("Upload your model file (.pth)", type=["pth"])

    if uploaded_file is not None:
        # Save uploaded file
        MODEL_PATH = os.path.join("models", uploaded_file.name)
        os.makedirs("models", exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ Model uploaded and saved as: {MODEL_PATH}")

# Try loading the model
if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        st.success(f"✅ Model loaded successfully from: {MODEL_PATH}")
    except Exception as e:
        st.error(f"❌ Error loading model weights: {e}")
        st.stop()
else:
    st.error("❌ No model available. Please upload `best_metric_model.pth` to proceed.")
    st.stop()

# Define transforms for inference (single image, grayscale)

transform = Compose([
    ScaleIntensity(),
    EnsureChannelFirst(),
])
