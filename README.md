# EEGFormer: Multi-Band EEG Transformer for Alzheimer's Detection

This project proposes a Multi-Band EEG Transformer for classifying EEG recordings into:

- Healthy Control
- Alzheimer's Disease
- Frontotemporal Dementia

## Research Idea

Instead of treating EEG as one raw signal, this model separates EEG into frequency bands:

- Delta
- Theta
- Alpha
- Beta
- Gamma

Each band is processed by a band-specific Transformer encoder, followed by cross-band attention.

## Project Flow

1. Load EEG `.set` files
2. Preprocess EEG signals
3. Apply band-pass filtering
4. Create fixed-size EEG windows
5. Train baseline models
6. Train proposed EEGFormer model
7. Evaluate using Accuracy, F1, AUC, Confusion Matrix
8. Perform ablation studies
9. Generate attention and SHAP visualizations

## Models

- Random Forest
- CNN
- LSTM
- Standard Transformer
- Proposed EEGFormer

## Dataset

OpenNeuro EEG dementia dataset: ds004504

## Goal

To test whether explicit modeling of EEG frequency-band interactions improves dementia classification.
