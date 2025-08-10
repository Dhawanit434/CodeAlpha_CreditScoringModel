# src/explain.py
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import shap
import matplotlib.pyplot as plt

ART = Path('artifacts')
X_train_proc = pd.read_csv(ART / 'X_train_processed.csv')
model = joblib.load(ART / 'best_model.joblib')

# use a background sample
bg = X_train_proc.sample(n=min(200, len(X_train_proc)), random_state=42)

# If model is tree-based, use TreeExplainer for speed
try:
    explainer = shap.Explainer(model, bg)
    shap_values = explainer(bg)
    # summary plot
    plt.figure(figsize=(8,6))
    shap.plots.bar(shap_values, max_display=25, show=False)
    plt.tight_layout()
    plt.savefig(ART / 'shap_summary.png', dpi=200)
    print("Saved SHAP summary to artifacts/shap_summary.png")
except Exception as e:
    print("SHAP explanation failed:", e)
    print("If using linear model, consider using coefficient-based feature importance.")

