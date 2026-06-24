# EEGFormer Results Summary

## Model Comparison

| Model | Input Type | Accuracy | Macro F1 |
|---|---|---:|---:|
| Random Forest | Statistical EEG features | 0.3749 | 0.3360 |
| CNN | Normalized raw EEG windows | 0.3894 | 0.3581 |
| LSTM | Normalized raw EEG windows | 0.4299 | 0.4090 |
| Transformer | Normalized raw EEG windows | 0.3920 | 0.3447 |
| EEGFormer | Normalized multi-band EEG windows | **0.7133** | **0.6896** |

## Key Finding

EEGFormer achieved the best performance with 71.33% accuracy and 0.6896 macro F1, substantially outperforming all baseline models.

## Interpretation

The results suggest that explicit frequency-band decomposition and cross-band modeling provide useful diagnostic information for dementia classification from EEG signals.

## EEGFormer Test Confusion Matrix

| Actual / Predicted | Alzheimer's | Healthy | FTD |
|---|---:|---:|---:|
| Alzheimer's | 749 | 164 | 104 |
| Healthy | 170 | 688 | 21 |
| FTD | 161 | 53 | 237 |