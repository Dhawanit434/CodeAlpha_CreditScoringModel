# src/app_streamlit.py
import streamlit as st
import joblib, json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

ART = Path('artifacts')
st.set_page_config(page_title="Credit Scoring Demo", layout='centered')

# load artifacts
preprocessor = joblib.load(ART / 'preprocessor.joblib')
model = joblib.load(ART / 'best_model.joblib')
meta = json.load(open(ART / 'metadata.json'))

st.title("Creditworthiness Predictor")
st.write("Fill applicant details and predict probability of being creditworthy.")

# dynamic input form
with st.form("input_form"):
    inputs = {}
    for col in meta['numeric_features']:
        default = meta['numeric_median'].get(col, 0.0)
        inputs[col] = st.number_input(col, value=float(default))

    for col in meta['categorical_features']:
        opts = meta['cat_unique_vals'].get(col, [])
        if len(opts) == 0:
            inputs[col] = st.text_input(col, value="")
        else:
            inputs[col] = st.selectbox(col, options=opts)

    submitted = st.form_submit_button("Predict")

if submitted:
    X_df = pd.DataFrame([inputs])
    # ensure same column order as original
    X_df = X_df.reindex(columns=meta['original_columns'])
    Xt = preprocessor.transform(X_df)
    prob = model.predict_proba(Xt)[:,1][0] if hasattr(model, 'predict_proba') else None
    pred = model.predict(Xt)[0]
    st.metric("Probability (positive class)", f"{prob:.3f}" if prob is not None else "N/A")
    st.write("Predicted label:", int(pred))

    # show feature importance if available
    if hasattr(model, 'feature_importances_'):
        feat_imp = pd.Series(model.feature_importances_, index=meta['feature_names']).sort_values(ascending=False).head(15)
        st.bar_chart(feat_imp)
    elif hasattr(model, 'coef_'):
        coefs = pd.Series(model.coef_.ravel(), index=meta['feature_names']).sort_values(key=abs, ascending=False).head(15)
        st.bar_chart(coefs)

    st.write("You can download model & preprocessor from the `artifacts/` folder in the repo.")
