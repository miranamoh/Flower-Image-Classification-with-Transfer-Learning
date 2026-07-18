# Flower-Image-Classification-with-Transfer-Learning
Flower species image classifier built with transfer learning on MobileNetV2, comparing baseline, dropout regularization, and fine-tuning strategies with 92% test accuracy.
<h1 align="center">🌸 Flower Image Classification with Transfer Learning (MobileNetV2)</h1>

<p align="center">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/MobileNetV2-Transfer_Learning-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Google_Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white"/>
</p>

<img width="557" height="467" alt="image" src="https://github.com/user-attachments/assets/b541e19e-4cc5-4eed-bff7-bdeaea3db460" />

---

##  Overview

An image classification pipeline that identifies flower species using **transfer learning** on **MobileNetV2**, trained and evaluated on the **TF Flowers dataset** (3,670 images across 5 classes: daisy, dandelion, roses, sunflowers, tulips). The project systematically compares a frozen-backbone baseline against progressively more advanced training strategies — stronger dropout regularization, selective fine-tuning, learning-rate scheduling, and architectural refinement — to study their effect on generalization.

### Problem It Solves

Training a CNN from scratch for image classification requires large labeled datasets and heavy compute. This project demonstrates how transfer learning from ImageNet-pretrained weights can achieve strong flower classification accuracy with a lightweight, mobile-friendly architecture (MobileNetV2), while systematically identifying which training strategy (regularization vs. fine-tuning vs. architecture changes) delivers the best generalization.

###  Project Goals

- Build a complete, reproducible image classification pipeline (loading → preprocessing → augmentation → training → evaluation)
- Compare a frozen-backbone baseline against dropout regularization and selective layer fine-tuning
- Apply learning-rate scheduling (`ReduceLROnPlateau`) and `EarlyStopping` to stabilize and optimize training
- Evaluate rigorously using accuracy, per-class precision/recall/F1, and confusion matrix analysis

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| Base Model | MobileNetV2 (ImageNet pretrained) |
| Dataset Loading | TensorFlow Datasets (`tfds`) — TF Flowers |
| Evaluation | scikit-learn (classification report, confusion matrix), seaborn |
| Environment | Jupyter Notebook / Google Colab (GPU runtime) |

---

##  Dataset

**TF Flowers** — 3,670 RGB images across 5 classes, loaded via `tensorflow_datasets`:

| Class | Images |
|---|---|
| Dandelion | 898 |
| Tulips | 799 |
| Sunflowers | 699 |
| Roses | 641 |
| Daisy | 633 |

Split (70/15/15, fixed seed 42): **2,569 training / 550 validation / 551 test** images.

**Preprocessing:** resize to 180×180, normalize pixel values to [0,1], with on-the-fly data augmentation applied to the training set via `tf.data` + `AUTOTUNE` prefetching.

---

##  Modeling Approach

Four progressive configurations were trained and compared:

| Stage | Configuration | Epochs | Val. Accuracy |
|---|---|---|---|
| **1. Baseline** | Frozen MobileNetV2 backbone + `GlobalAveragePooling2D` → `Dense(128, relu)` → `Dropout(0.3)` → `Dense(5, softmax)`, Adam lr=1e-3 | 20 | 87.27% |
| **2. Dropout Regularization** | Same architecture, `Dropout` increased to 0.5 | 20 | 87.82% |
| **3. Fine-Tuning** | Top **70 layers** of MobileNetV2 unfrozen, initialized from the Dropout model, Adam lr=1e-5 | 80 | **93.27%** |
| **4. Optimized (Callbacks)** | Continued fine-tuning with `EarlyStopping` (patience=3, restore best weights) and `ReduceLROnPlateau` (factor=0.5, patience=2, min_lr=1e-6) | up to +30 | **93.27%** (maintained, more stable) |

A final architectural refinement was then evaluated on the held-out **test set**: frozen backbone + `Dense(256, relu)` → `BatchNormalization` → `Dropout(0.3)` → `Dense(5, softmax)`, Adam lr=1e-4, 15 epochs.

**Final test performance:** **92% overall accuracy**, macro-average precision/recall/F1 = **0.92**. Dandelion scored highest (F1 = 0.94, most visually distinctive), while roses scored lowest (F1 = 0.88) due to visual similarity with tulips — confirmed by confusion matrix analysis, where roses↔tulips is the most frequent misclassification pair.

---

##  Project Structure

```
Flower-Image-Classification-with-Transfer-Learning/
├── flower_classification.ipynb   # Full pipeline: data loading → preprocessing → training → evaluation
├── .gitignore
└── README.md
```

---


## Usage

Open and run the notebook top to bottom (designed for Google Colab with a GPU runtime, but runs locally as well):

```bash
jupyter notebook flower_classification.ipynb
```

The notebook automatically downloads the TF Flowers dataset via `tensorflow_datasets` on first run — no manual dataset download needed.

---

##  Usage Example

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow import keras
from tensorflow.keras import layers

base_model = MobileNetV2(input_shape=(180, 180, 3), include_top=False, weights='imagenet')
base_model.trainable = False

inputs = keras.Input(shape=(180, 180, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(5, activation='softmax')(x)

model = keras.Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

---

##  Results

| Model | Validation Accuracy |
|---|---|
| Baseline | 87.27% |
| Baseline + Dropout 0.5 | 87.82% |
| Fine-Tuned (top-70 layers) | 93.27% |
| Optimized (+ callbacks) | 93.27% |

**Final test set (per-class):**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Dandelion | 0.91 | 0.98 | 0.94 |
| Daisy | 0.97 | 0.86 | 0.91 |
| Tulips | 0.93 | 0.90 | 0.91 |
| Sunflowers | 0.94 | 0.93 | 0.93 |
| Roses | 0.85 | 0.91 | 0.88 |

Overall test accuracy: **92%** (macro avg precision/recall/F1: 0.92)

---

## Core Contributions

My work on this project focused on the modeling side of the pipeline:
- Loading and configuring the pretrained MobileNetV2 backbone
- Building and training the baseline classification head
- Implementing and tuning Dropout regularization
- Hyperparameter tuning across training stages (learning rate, epochs, layer-unfreezing depth)

---

## Future Work

- Apply class-balanced sampling or focal loss to address the moderate class imbalance
- Experiment with alternative lightweight backbones (EfficientNetB0, MobileNetV3)
- Extend the dataset with additional real-world/web-scraped images for robustness
- Deploy the optimized model on a mobile/edge device to evaluate real-world inference latency

This project was developed for academic deep learning course. All rights reserved
