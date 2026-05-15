# Depth Estimation using U-Net++ on NYU Depth V2

## Overview

This project implements a **monocular depth estimation** pipeline using a deep learning approach based on **U-Net++** with a **ResNeXt50_32x4d encoder**. The model is trained on the **NYU Depth V2** dataset to predict dense depth maps from single RGB indoor images.

The project uses:

* **PyTorch**
* **Segmentation Models PyTorch (SMP)**
* **Albumentations**
* **Mixed Precision Training**
* **SSIM + MSE Evaluation**
* **Transfer Learning with ResNeXt50**

The model achieves strong qualitative and quantitative performance using a lightweight and clean training pipeline.

---

# Table of Contents

1. Project Goal
2. Dataset
3. Model Architecture
4. Training Strategy
5. Data Augmentation
6. Metrics
7. Loss Function
8. Mixed Precision Training
9. Project Structure
10. Installation
11. Usage
12. Training Pipeline
13. Inference Pipeline
14. Results
15. Visualization
16. Key Features
17. Future Improvements
18. License
19. Acknowledgment

---

# 1. Project Goal

The purpose of this project is to estimate scene depth from a single RGB image.

Depth estimation is an important computer vision task used in:

* Autonomous driving
* Robotics
* AR/VR
* 3D reconstruction
* Scene understanding
* SLAM systems

Unlike stereo vision systems, this project performs **monocular depth estimation**, meaning the model predicts depth using only one RGB image.

---

# 2. Dataset

## NYU Depth V2

The project uses the **NYU Depth V2** dataset, one of the most popular benchmarks for indoor depth estimation.

Dataset contains:

* RGB indoor images
* Corresponding depth maps
* Various indoor scenes:

  * Bedrooms
  * Kitchens
  * Offices
  * Living rooms
  * Bathrooms

### Dataset Split

| Split      | Samples |
| ---------- | ------- |
| Train      | 45,619  |
| Validation | 4,562   |
| Test       | 507     |

### Data Format

Each sample includes:

* RGB image
* Corresponding grayscale depth map

Depth maps are normalized to:

```python
depth = depth / 255.
```

---

# 3. Model Architecture

## U-Net++

The core model is based on **U-Net++**, an advanced variation of U-Net with redesigned skip pathways and dense skip connections.

### Why U-Net++?

Compared to standard U-Net:

* Better feature fusion
* Reduced semantic gap between encoder and decoder
* Improved multi-scale learning
* Better segmentation/regression performance

---

## Encoder

The encoder backbone is:

```python
resnext50_32x4d
```

### Benefits of ResNeXt50

* Strong feature extraction
* Efficient grouped convolutions
* Better representation learning
* Pretrained on ImageNet

---

## Final Model

```python
smp.UnetPlusPlus(
    encoder_name='resnext50_32x4d',
    in_channels=3,
    classes=1
)
```

### Input

* RGB image
* Shape: `(3, 224, 224)`

### Output

* Single-channel depth map
* Shape: `(1, 224, 224)`

---

# 4. Training Strategy

The project uses a **two-stage training strategy**.

## Stage 1 — Frozen Encoder

Initially, the encoder is frozen:

```python
model.trainable_encoder(trainable=False)
```

Only the decoder is trained.

### Why?

This helps:

* Stabilize training
* Preserve pretrained ImageNet features
* Allow decoder adaptation first

---

## Stage 2 — Full Fine-Tuning

After several epochs:

```python
model.trainable_encoder(trainable=True)
```

Both encoder and decoder are trained together.

This improves:

* Feature adaptation
* Domain learning
* Final accuracy

---

# 5. Data Augmentation

Extensive augmentations are applied using Albumentations.

## Applied Augmentations

### Geometric Transformations

* Horizontal Flip
* ShiftScaleRotate
* RandomResizedCrop

### Noise & Blur

* Gaussian Noise
* Motion Blur
* Median Blur
* Standard Blur

### Color Transformations

* RGB Shift
* Brightness/Contrast
* Color Jitter
* Hue/Saturation Adjustments

---

## Final Preprocessing

```python
A.Resize(224, 224)
A.Normalize()
ToTensorV2()
```

---

# 6. Metrics

The project evaluates model performance using:

---

## SSIM (Structural Similarity Index)

SSIM measures structural similarity between predicted and ground truth depth maps.

Higher is better.

### Advantages

* Perceptual quality aware
* Considers:

  * Luminance
  * Contrast
  * Structure

### Formula

SSIM(x,y)=\frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}{(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}

---

## MSE (Mean Squared Error)

Measures pixel-wise regression error.

Lower is better.

### Formula

MSE=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2

---

# 7. Loss Function

The project uses:

```python
nn.MSELoss()
```

Why MSE?

* Stable regression optimization
* Common for dense prediction tasks
* Easy to optimize

---

# 8. Mixed Precision Training

The project uses PyTorch AMP:

```python
from torch.cuda.amp import autocast, GradScaler
```

## Benefits

* Faster training
* Reduced GPU memory usage
* Larger effective batch sizes
* Improved throughput

---

# 9. Project Structure

```text
project/
│
├── train.py
├── README.md
├── requirements.txt
│
├── models/
│   └── nyu-v2-depth-resnext50_32x4d-unetplusplus.pt
│
├── outputs/
│   ├── predictions/
│   ├── logs/
│   └── visualizations/
│
└── dataset/
    └── nyu_depth_v2/
```

---

# 10. Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/depth-estimation-unetplusplus.git

cd depth-estimation-unetplusplus
```

---

## Install Dependencies

```bash
pip install -U segmentation-models-pytorch
```

Additional dependencies:

```bash
pip install torch torchvision albumentations matplotlib pandas tqdm opencv-python pillow torchmetrics scikit-learn
```

---

# 11. Usage

## Train Model

```python
python train.py
```

---

## Load Best Model

```python
best_sd = torch.load('nyu-v2-depth-resnext50_32x4d-unetplusplus.pt')
model.load_state_dict(best_sd)
```

---

## Run Inference

```python
preds = model(img)
```

---

# 12. Training Pipeline

## Training Steps

1. Load dataset
2. Apply augmentations
3. Create DataLoaders
4. Initialize model
5. Freeze encoder
6. Train decoder
7. Unfreeze encoder
8. Fine-tune full model
9. Save best checkpoint
10. Evaluate on test set

---

## Optimizer

```python
AdamW
```

### Configuration

```python
lr = 1e-3
weight_decay = 0.02
```

---

## Scheduler

```python
OneCycleLR
```

Advantages:

* Faster convergence
* Better generalization
* Dynamic learning rate scheduling

---

## Gradient Clipping

```python
nn.utils.clip_grad_norm_(...)
```

Used to stabilize training.

---

# 13. Inference Pipeline

During inference:

* Gradients are disabled
* Mixed precision is enabled
* Predictions are collected
* Metrics are computed
* Results are visualized

---

# 14. Results

## Final Metrics

| Metric          | Value  |
| --------------- | ------ |
| Validation SSIM | 0.909  |
| Validation MSE  | 0.0028 |

---

## Training Progress

| Epoch | Train Loss | Val Loss | Train SSIM | Val SSIM |
| ----- | ---------- | -------- | ---------- | -------- |
| 0     | 0.0953     | 0.0097   | 0.5750     | 0.7697   |
| 1     | 0.0102     | 0.0057   | 0.8415     | 0.8671   |
| 2     | 0.0104     | 0.0045   | 0.8724     | 0.8889   |
| 3     | 0.0068     | 0.0032   | 0.8979     | 0.9038   |
| 4     | 0.0050     | 0.0028   | 0.9101     | 0.9091   |

---

# 15. Visualization

The project visualizes:

* Original RGB image
* Ground truth depth map
* Predicted depth map

Depth maps use the:

```python
inferno
```

colormap where:

* Brighter colors indicate larger depth
* Darker colors indicate closer objects

---

# 16. Key Features

## Highlights

* U-Net++ architecture
* ResNeXt50 pretrained encoder
* Mixed precision training
* Extensive augmentations
* SSIM evaluation
* OneCycle learning rate scheduling
* Gradient clipping
* Encoder freezing/unfreezing
* Clean modular implementation

---

# 17. Future Improvements

Potential future enhancements:

## Architecture Improvements

* Attention U-Net++
* DeepLabV3+
* Swin Transformer backbone
* ConvNeXt encoder

---

## Loss Functions

Try combinations such as:

```python
L1 + SSIM
```

or:

```python
BerHu Loss
```

---

## Metrics

Additional metrics:

* RMSE
* MAE
* δ Accuracy thresholds

---

## Data Improvements

* Higher resolution training
* Better depth normalization
* Test-time augmentation

---

## Training Improvements

* Longer training
* Cosine annealing
* EMA weights
* Self-supervised pretraining

---

# 18. License

This project is open-source and available under the MIT License.

---

# 19. Acknowledgment

Libraries and frameworks used:

* PyTorch
* Segmentation Models PyTorch
* Albumentations
* TorchMetrics
* OpenCV
* NumPy
* Matplotlib

Dataset:

* NYU Depth V2

---

# Author

**Shayan Rokhva**

Email: `shayanrokhva1999@gmail.com`

---
# !!! ReadMe is generated by GPT. Check the important information !!!

