
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def detect_target(df):
    # heuristic: common names
    common = ['creditworthy','credit_worthy','credit_worthiness','creditworthy_flag',
              'default','loan_status','target','y','class','credit']
    for col in df.columns:
        if col.lower() in common:
            return col
    # binary columns
    binary = [c for c in df.columns if df[c].nunique() == 2]
    if len(binary) == 1:
        return binary[0]
    # columns containing keywords
    for col in df.columns:
        if any(k in col.lower() for k in ['credit','default','risk','status']):
            return col
    return None

def main(args):
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    print(f"Loaded {input_path} shape={df.shape}")
    print("Columns:", df.columns.tolist()[:50])

    target = args.target or detect_target(df)
    if target is None:
        raise ValueError("Target could not be auto-detected. Re-run with --target TARGET_COLUMN")

    print("Using target column:", target)

    # drop obvious ID columns
    id_cols = [c for c in df.columns if c.lower() == 'id' or c.lower().endswith('_id')]
    df = df.drop(columns=id_cols, errors='ignore')

    # Basic cleanup: drop rows missing target
    df = df.dropna(subset=[target]).reset_index(drop=True)

    # Split X/y early (we will fit preprocessor on train only)
    X = df.drop(columns=[target])
    y = df[target].astype(int) if pd.api.types.is_integer_dtype(df[target]) or df[target].dropna().nunique()<=2 else df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y if len(np.unique(y))>1 else None
    )

    # detect numeric/categorical — treat low-cardinality numerics as categorical
    numeric_feats = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_feats = [c for c in X_train.columns if c not in numeric_feats]

    for c in numeric_feats.copy():
        if X_train[c].nunique() <= 15:
            numeric_feats.remove(c)
            categorical_feats.append(c)

    print("Numeric features:", numeric_feats[:10])
    print("Categorical features:", categorical_feats[:10])

    # Numeric pipeline
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, numeric_feats),
        ('cat', cat_pipeline, categorical_feats)
    ])

    # fit preprocessor on train
    preprocessor.fit(X_train)

    # feature names after encoding
    num_names = numeric_feats
    if len(categorical_feats) > 0:
        ohe = preprocessor.named_transformers_['cat'].named_steps['ohe']
        cat_names = list(ohe.get_feature_names_out(categorical_feats))
    else:
        cat_names = []
    feature_names = num_names + cat_names

    # transform and save processed csvs
    X_train_tr = preprocessor.transform(X_train)
    X_test_tr = preprocessor.transform(X_test)

    X_train_df = pd.DataFrame(X_train_tr, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_tr, columns=feature_names)

    X_train_df.to_csv(out_dir / 'X_train_processed.csv', index=False)
    X_test_df.to_csv(out_dir / 'X_test_processed.csv', index=False)
    y_train.to_csv(out_dir / 'y_train.csv', index=False)
    y_test.to_csv(out_dir / 'y_test.csv', index=False)

    # metadata for Streamlit UI
    metadata = {
        'target': str(target),
        'numeric_features': numeric_feats,
        'categorical_features': categorical_feats,
        'feature_names': feature_names,
        'cat_unique_vals': {c: X_train[c].dropna().unique().tolist() for c in categorical_feats},
        'numeric_median': {c: float(X_train[c].median()) for c in numeric_feats},
        'numeric_mean': {c: float(X_train[c].mean()) for c in numeric_feats},
        'original_columns': X_train.columns.tolist()
    }

    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    joblib.dump(preprocessor, out_dir / 'preprocessor.joblib')
    print("Saved preprocessor and processed data to", out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='path to raw CSV')
    parser.add_argument('--out_dir', default='artifacts', help='artifact folder')
    parser.add_argument('--target', default=None, help='target column name (if auto-detect fails)')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()
    main(args)
