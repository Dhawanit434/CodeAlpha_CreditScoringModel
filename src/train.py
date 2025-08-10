# src/train.py
import argparse
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

def main(args):
    artifacts = Path(args.artifacts)
    X_train = pd.read_csv(artifacts / 'X_train_processed.csv')
    y_train = pd.read_csv(artifacts / 'y_train.csv', header=0).squeeze()
    models_dir = artifacts / 'models'
    models_dir.mkdir(exist_ok=True)

    results = []

    # 1) Logistic Regression (baseline)
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr.fit(X_train, y_train)
    results.append(('logreg', lr))

    # 2) Random Forest with small grid
    rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    rf_params = {'n_estimators':[100,200], 'max_depth':[None,10,20]}
    gs_rf = GridSearchCV(rf, rf_params, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    gs_rf.fit(X_train, y_train)
    results.append(('rf', gs_rf.best_estimator_))

    # 3) XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
    xgb_params = {'n_estimators':[100,200], 'max_depth':[3,6], 'learning_rate':[0.05,0.1]}
    gs_xgb = GridSearchCV(xgb, xgb_params, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    gs_xgb.fit(X_train, y_train)
    results.append(('xgb', gs_xgb.best_estimator_))

    # Evaluate models on training data (quick)
    metrics = {}
    for name, model in results:
        probs = model.predict_proba(X_train)[:,1] if hasattr(model, 'predict_proba') else model.decision_function(X_train)
        preds = model.predict(X_train)
        metrics[name] = {
            'train_f1': classification_report(y_train, preds, output_dict=True).get('weighted avg', {}).get('f1-score'),
            'train_rocauc': roc_auc_score(y_train, probs)
        }
        joblib.dump(model, models_dir / f'{name}.joblib')

    # pick best by train_rocauc (or f1) — here rocauc
    best_name = max(metrics.items(), key=lambda x: x[1]['train_rocauc'])[0]
    best_model = joblib.load(models_dir / f'{best_name}.joblib')
    joblib.dump(best_model, artifacts / 'best_model.joblib')

    # save metrics
    import json
    with open(artifacts / 'train_metrics.json','w') as f:
        json.dump(metrics, f, indent=2)

    print("Trained models saved to", models_dir)
    print("Best model:", best_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', default='artifacts', help='artifacts folder created by preprocessing')
    args = parser.parse_args()
    main(args)
