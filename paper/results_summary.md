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

## Ablation Study

| Experiment | Accuracy | Macro F1 |
|---|---:|---:|
| Delta Only | 0.4653 | 0.4533 |
| Theta Only | 0.5215 | 0.4536 |
| Alpha Only | 0.5190 | 0.4598 |
| Beta Only | 0.5070 | 0.4685 |
| Gamma Only | 0.4810 | 0.4833 |
| EEGFormer No Cross-Attention | 0.5709 | 0.5604 |
| Full EEGFormer | **0.7133** | **0.6896** |



## Ablation Study

| Experiment | Accuracy | Macro F1 |
|---|---:|---:|
| Delta Only | 0.4653 | 0.4533 |
| Theta Only | 0.5215 | 0.4536 |
| Alpha Only | 0.5190 | 0.4598 |
| Beta Only | 0.5070 | 0.4685 |
| Gamma Only | 0.4810 | 0.4833 |
| EEGFormer No Cross-Attention | 0.5709 | 0.5604 |
| Full EEGFormer | **0.7133** | **0.6896** |

## Ablation Interpretation

| Experiment | Accuracy | Macro F1 |
|---|---:|---:|
| Delta Only | 0.4653 | 0.4533 |
| Theta Only | 0.5215 | 0.4536 |
| Alpha Only | 0.5190 | 0.4598 |
| Beta Only | 0.5070 | 0.4685 |
| Gamma Only | 0.4810 | 0.4833 |
| EEGFormer No Cross-Attention | 0.5709 | 0.5604 |
| Full EEGFormer | **0.7133** | **0.6896** |

## Ablation Interpretation

No individual EEG frequency band achieved performance close to the full EEGFormer model. Removing cross-band attention reduced accuracy from 0.7133 to 0.5709.

This shows that both multi-band representation and cross-band attention are important for dementia classification.

## Model Complexity Analysis

| Model | Parameters | Inference Time (sec/sample) |
|-------|------------:|----------------------------:|
| CNN | 40,131 | 0.000429 |
| LSTM | 143,235 | 0.057870 |
| Transformer | 525,571 | 0.000823 |
| EEGFormer | 596,547 | 0.002490 |

EEGFormer contains the highest number of parameters among the evaluated models but still maintains low inference latency (~2.49 ms per sample) while achieving the best classification performance.

## ROC-AUC Results

| Class | ROC-AUC |
|---|---:|
| Alzheimer's | 0.7824 |
| Healthy Control | 0.8308 |
| FTD | 0.7978 |

## Precision-Recall Results

| Class | Average Precision |
|---|---:|
| Alzheimer's | 0.6341 |
| Healthy Control | 0.8340 |
| FTD | 0.5913 |

## Cross-Band Attention Interpretation

EEGFormer attention analysis showed strong cross-band interactions involving Gamma and Alpha activity.

The strongest interactions included:

| Query Band | Attended Band | Attention Weight |
|---|---|---:|
| Delta | Gamma | 0.456 |
| Alpha | Gamma | 0.436 |
| Beta | Gamma | 0.470 |
| Gamma | Alpha | 0.294 |

This suggests that EEGFormer does not treat frequency bands independently. Instead, it learns interactions between neural rhythms, supporting the proposed cross-band attention mechanism.