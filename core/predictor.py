import joblib
import pandas as pd
import streamlit as st

@st.cache_resource
def load_model(model_path="models/bodyfat_xgboost_model_final.pkl"):
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Lỗi load model: {e}")
        return None

def predict_body_fat(model, data_dict):
    input_df = pd.DataFrame([data_dict])
    prediction = float(model.predict(input_df)[0])
    return prediction