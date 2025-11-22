# Heart Disease Prediction Models

This directory contains trained Transformer models for heart disease prediction.

## Files:
- preprocessing_objects.pkl: Scaler and label encoders for data preprocessing
- model_configs.json: Model architecture configurations
- *_model.pth: Saved PyTorch model weights
- performance_metrics.pkl: Model performance statistics

## Model Types:
1. Binary Classification: Predicts presence/absence of heart disease
2. Multiclass Classification: Predicts disease severity levels (0-4)

## Sampling Techniques:
- No Sampling: Original data distribution
- SMOTE: Synthetic Minority Over-sampling Technique
- SMOTETomek: Combined over-sampling and under-sampling

## Usage:
Use the provided Streamlit app (streamlit_app.py) to make predictions with these models.
