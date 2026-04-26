# FraudBERT: LLM-Enhanced Categorical Feature Encoding for Insurance Fraud Detection

**Author:** Juharasha Shaik — Independent Researcher  
**Contact:** shaik.juharasha@ieee.org

## Overview

FraudBERT is a hybrid fraud detection framework that uses pre-trained language model embeddings to encode categorical insurance features, then fuses them with numerical features for downstream ML classifiers. This repository contains the full reproducible experiment pipeline.

**Datasets:**
- [Vehicle Claim Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection) (Kaggle: shivamb)
- [Insurance Claims Fraud](https://www.kaggle.com/datasets/mastmustu/insurance-claims-fraud-data) (Kaggle: mastmustu)

## Key Results

FraudBERT consistently outperforms all baselines (Logistic Regression, Random Forest, XGBoost, LightGBM, MLP, Transformer) across AUPRC, AUC-ROC, F1, and Recall@95%Precision on both datasets.

## Models Compared

| Model | Type |
|-------|------|
| Logistic Regression | Classical baseline |
| Random Forest | Classical baseline |
| XGBoost | Gradient boosting baseline |
| LightGBM | Gradient boosting baseline |
| MLP (3-layer) | Deep learning baseline |
| Transformer | TabTransformer-style baseline |
| **FraudBERT + XGBoost** | **Proposed** |
| **FraudBERT + LightGBM** | **Proposed** |
| **FraudBERT + MLP** | **Proposed (learnable fusion)** |

## Repository Structure

```
paper1_fraudbert/
├── code/
│   ├── step1_encode.py        # LLM embedding generation (run first)
│   ├── step2_train.py         # Model training + evaluation
│   ├── models.py              # MLPClassifier, TransformerClassifier, FraudBERTMLP
│   ├── ablations.py           # Ablation studies (dim, encoder, imbalance)
│   ├── test_pipeline.py       # Smoke test with synthetic data
│   └── setup_and_download.sh  # Kaggle dataset downloader
├── data/
│   └── embeddings/            # Pre-computed .npy files (gitignored, regenerate locally)
├── results/                   # JSON results + AUPRC bar charts
├── requirements.txt
└── paper1_draft.md            # Full paper draft
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Smoke test (no data download needed)
python3 code/test_pipeline.py

# 3. Download real datasets (requires Kaggle API key at ~/.kaggle/kaggle.json)
bash code/setup_and_download.sh

# 4. Generate LLM embeddings
python3 code/step1_encode.py --dataset vehicle
python3 code/step1_encode.py --dataset claims

# 5. Train all models + generate results tables
python3 code/step2_train.py --dataset vehicle
python3 code/step2_train.py --dataset claims

# 6. Run ablation studies
python3 code/ablations.py --dataset vehicle
```

## Method

FraudBERT encodes each categorical feature value as a natural-language prompt:

```
"Transaction feature PolicyType: Collision / Liability"
"Transaction feature VehicleCategory: Sport"
```

All unique (feature, value) pairs are encoded in a single batch call to `sentence-transformers/all-MiniLM-L6-v2`, producing 384-dimensional semantic embeddings. Per-feature embeddings are concatenated and projected via PCA (→128d), then fused with scaled numerical features before passing to the classifier head.

For FraudBERT+MLP, a learnable linear projection replaces PCA, enabling end-to-end fine-tuning of the fusion.

## Citation

If you use this code, please cite:

```bibtex
@article{shaik2025fraudbert,
  title={FraudBERT: LLM-Enhanced Categorical Feature Encoding for Insurance Fraud Detection},
  author={Shaik, Juharasha},
  journal={Expert Systems with Applications},
  year={2025}
}
```

## License

MIT License
