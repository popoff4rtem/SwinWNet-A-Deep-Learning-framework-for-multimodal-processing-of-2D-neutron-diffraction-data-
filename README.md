# SwinWNet  
**A Deep Learning Framework for Multimodal Processing of 2D Neutron Diffraction Data**

---

## 📌 Overview

This repository contains the official implementation of **SwinWNet** — a multimodal deep learning framework for **joint segmentation and super-resolution (upscaling)** of 2D time-of-flight neutron diffraction data acquired with position-sensitive detectors (PSD).

📄 (Paper link will be added after publication)

The framework is specifically designed for **physically safe processing** of diffraction patterns, preserving:
- peak positions,
- integral intensities,
- peak shapes,
- and overall diffraction physics.

![2D Diffractions Postpocessing samples](Diffraction%20framework%20PRX%20Intellegence/Inference_2d.png)

![Postpocessing samples in the space of interplanar distances I(d)](Diffraction%20framework%20PRX%20Intellegence/Inference_1d.png)

In addition to standard image-based metrics, the framework provides **physics-aware evaluation tools** operating directly in *d-space*.

The implementation accompanies the corresponding research article and includes:
- full training pipelines,
- pretrained models,
- evaluation scripts,
- physical distortion metrics,
- and experimental notebooks.

---

## ✨ Key Features

- **Multimodal input**: diffraction data + corresponding error matrix
- **Joint learning** of segmentation and super-resolution
- **Transformer-based architecture** (Swin Transformer backbone)
- **Physically motivated evaluation metrics**
- **Three-stage training strategy**
- Optional **reinforcement learning fine-tuning** (experimental)
- Ready-to-use pretrained models

---

## 🧠 Model Architecture

**SwinWNet** is a dual-branch architecture consisting of:
- a **segmentation branch** (SwinUNet-like),
- a **super-resolution (upscaling) branch**,
- and shared cross-scale representations.

![SwinWNet Architecture](Diffraction%20framework%20PRX%20Intellegence/SwinWNet%20architecture.png)

Both branches are built on **Swin Transformer blocks**, enabling efficient local self-attention while preserving spatial structure.

The model supports:
- single-channel input: diffraction only `[B,1,H,W]`,
- multimodal input: diffraction + error matrix `[B,2,H,W]`.

![SwinWNet Inference Pipline](Diffraction%20framework%20PRX%20Intellegence/SwinWNet%20Inference%20Procession.png)

---

## 🗂️ Repository Structure

### 📁 `datasets/`
Datasets used for training and evaluation.

- `segmentation_maps.pkl`  
  Ground truth segmentation masks for all crystal structures.

- `dataset.pkl`  
  Main training dataset.  
  📥 Available at:  
  https://huggingface.co/datasets/popoff4rtem/2D-Neutron-Diffraction-Dataset-for-Discriminative-Models

- `test_data.pkl`  
  Lightweight test dataset.  
  Usage instructions are provided in `tutorial.ipynb`.

- `test_diffraction+error_matrices.pt`  
  Same test data organized in tensor format.

---

### 📁 `experiments/`
Experimental notebooks used in the study.

- `transformer_segmentation.ipynb`  
  Pretraining of the segmentation branch.

- `SwinSR_train.ipynb`  
  Pretraining of the upscaling branch.

- `SwinWNet_train.ipynb`  
  Training SwinWNet using diffraction-only input.

- `SwinWNet_error_mx_file_tune.ipynb`  
  Full multimodal training (diffraction + error matrix).

- `Physycal_metrics_test.ipynb`  
  Evaluation of physical distortions in diffraction peaks.

- `SwinWNet_RL_fine_tune_updated.ipynb`  
  Reinforcement learning fine-tuning (experimental).

- `SwinUNet.py`, `SwinUNet_old.py`  
  Model architecture definitions.

---

### 📁 `models/`
Pretrained model weights.

- `SwinUnet_binary_segmentation_diffraction.pth`  
  Segmentation-only model.

- `SwinUnetSR_upscaler_for_segmented_diffraction.pth`  
  Upscaling-only model.

- `SwinWNet_diffraction.pth`  
  Full model (diffraction only).

- `SwinWNet_diffraction+error_matrix.pth`  
  Full multimodal model.

- `SwinWNet_finetuned_rl_simple_alpha_policy.pth`  
  RL-finetuned model (beta).

---

### 📁 `results/`
Stored evaluation metrics for all experimental scenarios.

---

### 📁 `support_files/`
Utilities for dataset generation and physical metric computation.

---

## ⚙️ Core Framework Files

- `SwinWNet.py`  
  Main model implementation.

- `Segmentator_pretrain.py`  
  Segmentation branch pretraining pipeline.

- `Upscaler_pretrain.py`  
  Upscaling branch pretraining pipeline.

- `FullModel_supervised_trainer.py`  
  Full supervised training pipeline.

- `Supervised_train_full_pipline.py`  
  Simplified end-to-end training pipeline.

- `ST_Inference_Pipline.py`  
  Supervised inference pipeline.

- `Diffraction_metrics.py`  
  Physics-aware peak distortion metrics.

---

### 🧪 Reinforcement Learning (Experimental)

- `RL_policy.py`  
  Alpha policy network.

- `RL_finetuning_pipline.py`  
  RL fine-tuning pipeline.

- `RL_Inference_Pipline.py`  
  RL-aware inference pipeline.

⚠️ **Note:** RL components are experimental and provided for research purposes.

---

## 📊 Evaluation Metrics

### Segmentation
- Pixel Accuracy
- IoU
- Dice
- Precision
- Recall  
Evaluated at thresholds **0.25 / 0.5 / 0.75**

### Super-Resolution
- PSNR
- SSIM

### Physical Metrics (d-space)
- Integral intensity distortion
- Peak intensity distortion
- Peak shape divergence (Wasserstein-1 distance)

---

## 🖥️ GUI Applications

This framework includes **two user-friendly graphical applications** that make working with 2D neutron diffraction data much easier and more intuitive.

### 🎯 DiffractionLabeler — Diffraction Pattern Labeling

An intuitive application for fast and accurate peak labeling on 2D diffraction images.

**How it works:**
- You select peaks on the 1D intensity profiles **I(d)** (interplanar spacing space).
- A special algorithm automatically projects the selected peaks back onto the 2D diffraction pattern.
- Result — clean **binary masks**.

**Features:**
- Load diffraction patterns in `.npy` format
- Full control over convolution parameters: scattering angle range, wavelengths, and d-scale discretization step
- Real-time mask preview and editing. To select a specific item, simply right-click and drag it to select it. You can deselect it by left-clicking.

**For quick testing** use the file:  
`datasets/diffractions.npy`

---

### 🤖 InferenceGUI — Model Inference & Visualization

A powerful application for running the **SwinWNet** model and exploring every stage of post-processing in detail.

**What you see after inference:**
- All internal processing stages (top panel)
- For each stage — the corresponding **I(d)** convolution with its error matrix
- Click the legend to hide/show individual convolutions
- Click a stage again to collapse its visualization

**Features:**
- Load diffraction patterns in `.npy` format
- Full control over diffraction and convolution parameters (same as in DiffractionLabeler)
- Automatic error matrix calculation
- Logarithmic viewing mode
- Pattern normalization
- Save results

**Example files for testing:**
`datasets/Al2O3_sapphire_diffraction.npy`
`datasets/C_graphite_diffraction.npy`
`datasets/Na2Ca3Al2F14_diffraction.npy`
`datasets/Rb_diffraction.npy`
`datasets/Si_diffraction.npy`
`datasets/UO2_diffraction.npy`

**Recommended model weights:**  
`models/SwinWNet_diffraction+error_matrix.pth`

---

## 🚀 Quick Start

### 1️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 2️⃣ Download dataset
```
from datasets import load_dataset
dataset = load_dataset("popoff4rtem/2D-Neutron-Diffraction-Dataset-for-Discriminative-Models")
```

### 3️⃣ Run tutorial
Open and follow:
tutorial.ipynb

### 🔬 Intended Use
This framework is designed for:

* neutron diffraction data post-processing
* physics-preserving super-resolution
* detector resolution enhancement
* uncertainty-aware diffraction analysis

It does not hallucinate physical features and can be safely applied in experimental workflows.

### 📖 Citation
If you use this code in your research, please cite the corresponding article:
```
bibtex@article{Popoff2025SwinWNet,
  title   = {SwinWnet: a multimodal dual-branch Swin Transformer framework for 2D neutron diffraction postprocessing},
  authors  = {Popov A.I., Novoselov I.E., Antropov N.O., Smirnov A.A., Kravtsov E.A., Ogorodnikov I.N.},
  journal = {PRX Intelligence},
  year    = {2025}
}
```

### ⚠️ Disclaimer
This project is provided for research purposes only.
Reinforcement learning components are experimental.

### 🤝 Acknowledgements
This work builds upon advances in:

* Transformer-based vision models
* super-resolution
* physics-aware machine learning


### 📬 Contact
For questions or collaboration:
Artem Popov popoff4rtem@gmail.com
