# Pneumonia Detection from Chest X-ray

This Streamlit app uses a deep learning model (DenseNet, trained with MONAI) to classify chest X-ray images as **Normal** or **Pneumonia**.

## Features

- Upload your chest X-ray image (JPG, PNG)
- Model predicts and shows confidence scores for each class
- Simple, user-friendly interface

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/shivesh2334-ai/Pneumonia-prediction-xray.git
cd Pneumonia-prediction-xray
```

### 2. Install dependencies

It's recommended to use a virtual environment:

```bash
pip install -r requirements.txt
```

### 3. Place your trained model

Copy your trained PyTorch model file (`best_metric_model.pth`) into the repository folder.

### 4. Run the app

```bash
streamlit run app.py
```

## Model Training

The app expects a MONAI-trained DenseNet model with:
- `spatial_dims=2`
- `in_channels=1`
- `out_channels=2`

If you need help with training, see [MONAI documentation](https://monai.io/).

## Notes

- This app is for demonstration purposes. For clinical use, validation on large datasets is required.
- Requires Python 3.8+.
