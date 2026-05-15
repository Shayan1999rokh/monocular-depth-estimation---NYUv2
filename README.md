# Depth Estimation with U-Net++ on NYU Depth V2

A clean and efficient deep learning pipeline for **monocular depth estimation** using **U-Net++** with a **ResNeXt50_32x4d encoder** on the **NYU Depth V2** dataset.
This project uses **PyTorch**, **segmentation_models_pytorch**, **Albumentations**, and **mixed precision training** to achieve high-quality depth prediction.

---

# Overview

This project predicts dense depth maps from single RGB indoor images using a U-Net++ architecture.

The model was trained on the NYU Depth V2 dataset and achieved:

| Metric | Validation Score |
| ------ | ---------------- |
| SSIM   | **0.909**        |
| MSE    | **0.0028**       |

The implementation includes:

* U-Net++ architecture
* Transfer learning with pretrained ResNeXt50 encoder
* Mixed precision training (AMP)
* Extensive data augmentation
* SSIM + MSE evaluation
* Visualization utilities
* Training/validation/test pipeline

---

# Model Architecture

The network is based on:

* **Architecture:** U-Net++
* **Encoder Backbone:** ResNeXt50_32x4d
* **Input:** RGB image
* **Output:** Single-channel depth map

Implemented using:

```python
segmentation_models_pytorch.UnetPlusPlus
```

---

# Dataset

Dataset used:

* NYU Depth V2

The dataset contains paired:

* Indoor RGB images
* Ground-truth depth maps

Dataset split:

| Split      | Samples |
| ---------- | ------- |
| Train      | 45,619  |
| Validation | 4,562   |
| Test       | 507     |

---

# Features

## Training Features

* Mixed precision training using `torch.cuda.amp`
* Gradient clipping
* OneCycleLR scheduler
* AdamW optimizer
* Encoder freezing/unfreezing strategy
* GPU memory optimization

## Augmentations

Implemented using Albumentations:

* Horizontal Flip
* Gaussian Noise
* Motion Blur
* Median Blur
* RGB Shift
* Random Brightness & Contrast
* RandomResizedCrop
* ColorJitter
* ShiftScaleRotate
* HueSaturationValue

---

# Evaluation Metrics

The project uses:

## Structural Similarity Index (SSIM)

SSIM measures perceptual similarity between predicted and ground-truth depth maps.

Higher is better.

## Mean Squared Error (MSE)

Measures pixel-wise reconstruction error.

Lower is better.

---

# Training Configuration

| Parameter     | Value      |
| ------------- | ---------- |
| Epochs        | 5          |
| Freeze Epochs | 2          |
| Batch Size    | 64         |
| Learning Rate | 1e-3       |
| Optimizer     | AdamW      |
| Scheduler     | OneCycleLR |
| Loss Function | MSELoss    |
| Image Size    | 224×224    |

---

# Results

## Training Logs

| Epoch | Train Loss | Val Loss | Train SSIM | Val SSIM | Train MSE | Val MSE |
| ----- | ---------- | -------- | ---------- | -------- | --------- | ------- |
| 0     | 0.0953     | 0.0097   | 0.5750     | 0.7697   | 0.0954    | 0.0097  |
| 1     | 0.0102     | 0.0057   | 0.8415     | 0.8671   | 0.0102    | 0.0058  |
| 2     | 0.0104     | 0.0045   | 0.8724     | 0.8889   | 0.0104    | 0.0046  |
| 3     | 0.0068     | 0.0032   | 0.8979     | 0.9038   | 0.0068    | 0.0032  |
| 4     | 0.0050     | 0.0028   | 0.9101     | 0.9091   | 0.0050    | 0.0028  |

---

# Visualization

The project includes utilities for:

* RGB image visualization
* Ground-truth depth map visualization
* Predicted depth map visualization
* Colored depth rendering using Inferno colormap

Brighter colors indicate greater depth.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/depth-estimation-unetplusplus.git
cd depth-estimation-unetplusplus
```

Install dependencies:

```bash
pip install -U segmentation-models-pytorch
pip install albumentations
pip install torchmetrics
```

---

# Required Libraries

```python
torch
torchvision
segmentation-models-pytorch
albumentations
opencv-python
numpy
pandas
matplotlib
scikit-learn
torchmetrics
Pillow
tqdm
```

---

# Training

Run the training notebook or script:

```bash
python train.py
```

or open the Kaggle notebook directly.

---

# Inference

Load trained weights:

```python
model.load_state_dict(torch.load('nyu-v2-depth-resnext50_32x4d-unetplusplus.pt'))
```

Run prediction:

```python
preds = model(images)
```

---

# Sample Pipeline

1. Load RGB image
2. Apply preprocessing & augmentations
3. Feed image into U-Net++
4. Predict depth map
5. Evaluate using SSIM and MSE
6. Visualize prediction

---

# Project Structure

```text
├── train.py
├── inference.py
├── README.md
├── requirements.txt
├── models/
├── outputs/
├── notebooks/
└── weights/
```

---

# Future Improvements

Possible future extensions:

* Train with higher input resolution
* Use transformer-based encoders
* Add attention mechanisms
* Experiment with depth-specific loss functions
* Multi-scale supervision
* Real-time inference optimization

---

# Acknowledgements

* NYU Depth V2 Dataset
* PyTorch
* segmentation_models_pytorch
* Albumentations

---

# License

This project is open-source and available under the MIT License.
