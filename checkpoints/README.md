# Model Weights and Checkpoints

This directory contains the trained neural network model weights for Group 1's Skin Lesion Classification project.

## Included Checkpoints
- `cag_mm_novel_best.pth` (~5.5 MB): The fully trained PyTorch checkpoint for the proposed **CAG-MM Fusion** model, including:
  - Cross-Paradigm Self-Attention (CPSA) transformer weights
  - Dynamic Adaptive Gating (DAG) network weights
  - Multimodal clinical metadata projection weights
  - 8-class classification head weights trained with Cost-Sensitive Clinical Focal Loss.

## Full Backbone Weights (Cloud Storage & Kaggle)
Due to standard course submission zip size limitations, the full individual fine-tuned backbones are hosted and permanently accessible through our Kaggle cloud GPU execution runs:
1. `mobilenetv3_best.pth` (~17 MB)
2. `fasternet_best.pth` (~25 MB)
3. `shvit_best.pth` (~65 MB)

**Access & Download Link:**
- Base Model Kaggle Output: [https://www.kaggle.com/code/randomaayush/isic-2019-late-fusion-base-model/output](https://www.kaggle.com/code/randomaayush/isic-2019-late-fusion-base-model/output)
- Novelty Model Kaggle Output: [https://www.kaggle.com/code/randomaayush/isic-2019-novel-cag-mm-fusion-model/output](https://www.kaggle.com/code/randomaayush/isic-2019-novel-cag-mm-fusion-model/output)

