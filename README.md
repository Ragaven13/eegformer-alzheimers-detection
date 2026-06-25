<div align="center">

# 🧠 EEGFormer
### Multi-Band Cross-Attention Transformer for EEG-Based Dementia Classification

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![IEEE](https://img.shields.io/badge/Research-IEEE-orange.svg)]()
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

</div>

---

# 📌 Overview

EEGFormer is a deep learning framework for automated dementia classification from resting-state EEG recordings.

The model decomposes EEG signals into multiple frequency bands and learns interactions between them using a cross-band attention mechanism.

The framework performs multi-class classification of:

- Alzheimer's Disease (AD)
- Healthy Control (HC)
- Frontotemporal Dementia (FTD)

---

# 🧠 Motivation

Neurological disorders alter neural oscillatory activity across multiple EEG frequency bands.

Most existing methods:

- analyze frequency bands independently
- use handcrafted features
- fail to model interactions between neural rhythms

EEGFormer addresses these limitations by learning:

```
Delta ↔ Theta ↔ Alpha ↔ Beta ↔ Gamma
```

relationships directly from raw EEG windows.

---

# 🏗 Architecture

```
Raw EEG
   │
   ├── Sliding Windows
   │
   ├── Band Decomposition
   │      ├── Delta
   │      ├── Theta
   │      ├── Alpha
   │      ├── Beta
   │      └── Gamma
   │
   ├── CNN Band Encoders
   │
   ├── Cross-Band Attention
   │
   ├── Feature Fusion
   │
   └── Dementia Classification
```

---

# 📂 Dataset

### Subjects

| Group | Subjects |
|-------|----------:|
| Alzheimer's Disease | 36 |
| Healthy Control | 29 |
| Frontotemporal Dementia | 23 |
| Total | 88 |

---

### EEG Configuration

| Property | Value |
|----------|--------|
| Channels | 19 |
| Sampling Rate | 500 Hz |
| Window Length | 5 seconds |
| Samples per Window | 2500 |
| Frequency Bands | 5 |

---

# 🌊 Frequency Bands

| Band | Range |
|------|--------|
| Delta | 0.5 – 4 Hz |
| Theta | 4 – 8 Hz |
| Alpha | 8 – 13 Hz |
| Beta | 13 – 30 Hz |
| Gamma | 30 – 45 Hz |

---

# 📁 Project Structure

```text
EEGFormer
│
├── data
│   ├── raw
│   └── processed
│
├── src
│   ├── preprocessing
│   ├── data
│   ├── models
│   ├── training
│   └── evaluation
│
├── results
│   ├── checkpoints
│   ├── figures
│   ├── history
│   └── tables
│
├── paper
│
└── README.md
```

---

# 🚀 Installation

```bash
git clone https://github.com/YOUR_USERNAME/EEGFormer.git

cd EEGFormer

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

# ⚙️ Dataset Construction

Build EEG windows:

```bash
python src/preprocessing/build_window_dataset.py
```

Build multi-band windows:

```bash
python src/preprocessing/build_band_window_dataset.py
```

---

# 🏃 Training

### CNN

```bash
python src/training/train_cnn.py
```

### LSTM

```bash
python src/training/train_lstm.py
```

### Transformer

```bash
python src/training/train_transformer.py
```

### EEGFormer

```bash
python src/training/train_eegformer.py
```

---

# 📊 Results

## Baselines

| Model | Accuracy | Macro F1 |
|-------|----------:|----------:|
| Random Forest | 0.4333 | 0.2015 |
| CNN | 0.3894 | 0.3581 |
| LSTM | 0.4299 | 0.4090 |
| Transformer | 0.3920 | 0.3447 |
| EEGFormer | **0.7133** | **0.6896** |

---

# 🔬 Ablation Study

| Experiment | Accuracy | Macro F1 |
|------------|----------:|----------:|
| Delta Only | 0.4653 | 0.4533 |
| Theta Only | 0.5215 | 0.4536 |
| Alpha Only | 0.5190 | 0.4598 |
| Beta Only | 0.5070 | 0.4685 |
| Gamma Only | 0.4810 | 0.4833 |
| EEGFormer (No Cross Attention) | 0.5709 | 0.5604 |
| EEGFormer | **0.7133** | **0.6896** |

---

# 📈 ROC-AUC

| Class | ROC-AUC |
|-------|---------:|
| Alzheimer's | 0.7824 |
| Healthy Control | 0.8308 |
| FTD | 0.7978 |

---

# 📈 Average Precision

| Class | AP |
|-------|---:|
| Alzheimer's | 0.6341 |
| Healthy Control | 0.8340 |
| FTD | 0.5913 |

---

# ⚡ Model Complexity

| Model | Parameters | Inference Time |
|-------|------------:|----------------:|
| CNN | 40,131 | 0.000429 sec |
| LSTM | 143,235 | 0.057870 sec |
| Transformer | 525,571 | 0.000823 sec |
| EEGFormer | 596,547 | 0.002490 sec |

---

# 🧠 Cross-Band Attention

The learned attention matrix revealed strong interactions involving Gamma activity:

- Delta → Gamma : 0.456
- Alpha → Gamma : 0.436
- Beta → Gamma : 0.470
- Gamma → Alpha : 0.294

These findings suggest that dementia-related EEG signatures emerge from interactions between neural oscillations rather than isolated frequency bands.

---

# 📷 Figures

Place generated figures inside:

```text
results/figures/
```

Examples:

- training_loss.png
- training_accuracy.png
- training_f1.png
- eegformer_roc_curve.png
- eegformer_precision_recall_curve.png
- cross_band_attention_heatmap.png
- eegformer_ablation_study.png

---

# 🔁 Reproducibility

```bash
python src/training/train_eegformer.py

python src/evaluation/evaluate_eegformer_roc.py

python src/evaluation/evaluate_eegformer_pr_curve.py

python src/evaluation/plot_cross_band_attention.py
```

---

# 📚 Citation

```bibtex
@article{eegformer2026,
  title={EEGFormer: Multi-Band Cross-Attention Transformer for EEG-Based Dementia Classification},
  author={Anandha Ragaven Ravi},
  year={2026}
}
```

---

# ⭐ If you find this work useful, please consider starring the repository.
