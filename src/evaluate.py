# src/evaluate.py
import argparse
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

def main(args):
    artifacts = Path(args.artifacts)
    X_test = pd.read_csv(artifacts / 'X_test_processed.csv')
    y_test = pd.read_csv(artifacts / 'y_test.csv', header=0).squeeze()
    model = joblib.load(artifacts / 'best_model.joblib')

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)

    report = classification_report(y_test, preds, output_dict=True)
    roc = roc_auc_score(y_test, probs)

    with open(artifacts / 'test_metrics.json','w') as f:
        json.dump({'classification_report': report, 'roc_auc': roc}, f, indent=2)

    # confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix')
    plt.savefig(artifacts / 'confusion_matrix.png', bbox_inches='tight')
    print("Evaluation complete. ROC AUC:", roc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', default='artifacts')
    args = parser.parse_args()
    main(args)
