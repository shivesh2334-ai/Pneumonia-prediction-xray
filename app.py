import streamlit as st
import torch
import numpy as np
from monai.transforms import Compose, ScaleIntensity, EnsureChannelFirst, ToTensor
from monai.networks.nets import DenseNet
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = DenseNet(spatial_dims=2, in_channels=1, out_channels=2).to(device)
model.load_state_dict(torch.load("best_metric_model.pth", map_location=device))
model.eval()

# Define transforms for inference (single image, grayscale)
val_transforms = Compose([
    ScaleIntensity(),   # scale pixel values to [0, 1]
    AddChannel(),       # add channel dimension for grayscale
    ToTensor()          # convert to torch tensor
])

class_names = ['Normal', 'Pneumonia']

st.title("Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image to classify it as Normal or Pneumonia.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L") # Convert to grayscale

    try:
        # Convert image to numpy array
        np_image = np.array(image).astype(np.float32)
        # Apply transforms
        input_tensor = val_transforms(np_image)
        input_tensor = input_tensor.unsqueeze(0).to(device)  # add batch dim

        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]

        predicted_class_idx = torch.argmax(probabilities).item()
        confidence_normal = probabilities[0].item()
        confidence_pneumonia = probabilities[1].item()
        predicted_class_name = class_names[predicted_class_idx]

        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write(f"Prediction: **{predicted_class_name}**")
        st.write(f"Confidence (Normal): {confidence_normal:.4f}")
        st.write(f"Confidence (Pneumonia): {confidence_pneumonia:.4f}")

        if max(confidence_normal, confidence_pneumonia) < 0.7:
            st.warning("Prediction confidence is low. Please use high-quality images.")

    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.text(f"{e}")
