# Fraud-Detection

Practical ML class final project on fraud detection, using the
[Bank Account Fraud Dataset (NeurIPS 2022)](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)
from Kaggle.

## Project structure

```
Fraud-Detection/
├── .github/workflows/ci.yml   # CI: lint + tests on every PR
├── data/                      # gitignored datasets (regenerated via loader)
├── notebooks/                 # exploration; kept out of the import path
├── configs/                   # experiment / model config (yaml)
├── src/fraud_detection/       # importable package (add __init__.py)
│   ├── data/                  # loading / splitting 
│   ├── features/              # feature engineering
│   ├── models/                # train / evaluate / predict
│   └── services/              # entrypoint scripts
├── tests/                     # pytest tests mirroring src/
├── pyproject.toml
└── uv.lock
```
