# Tourism Purchase Predictor (RandomForest)

This repository contains a tuned RandomForestClassifier for predicting `ProdTaken` (purchase of the tourism package).

- Dataset: https://huggingface.co/datasets/pratapgangupaenta/tourism-dataset
- Selection metric: ROC AUC (5-fold CV)
- Best CV ROC AUC: 0.9513

## Inference (Python)
```python
import joblib
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="pratapgangupaenta/tourism-purchase-predictor-rf", filename="best_model.joblib")
model = joblib.load(model_path)
# model is a sklearn Pipeline: model.predict(X) or model.predict_proba(X)
```