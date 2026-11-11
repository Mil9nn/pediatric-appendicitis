"""Model loading and prediction helpers."""

import joblib
import pandas as pd
import streamlit as st


@st.cache_resource
def load_models():
    """Load both trained pipelines once and reuse them across reruns."""
    clinical_model = joblib.load("models/appendicitis_rf_pipeline.joblib")
    ultrasound_model = joblib.load("models/appendicitis_rf_us_pipeline.joblib")
    return clinical_model, ultrasound_model


def predict_risk(values, use_ultrasound, clinical_model, ultrasound_model):
    """Build model input in training-feature order and return prediction details."""
    model = ultrasound_model if use_ultrasound else clinical_model
    input_df = pd.DataFrame([values])[list(model.feature_names_in_)]
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    return prediction, probability
