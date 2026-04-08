import joblib
import pandas as pd
import streamlit as st
import os

@st.cache_resource
def load_model_v5(model_path="models/bodyfat_ai_super_clean_v5.pkl"):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"Không tìm thấy model: {model_path}")
        return None


def predict_body_fat_v5(model, raw_data_dict):

    h = float(raw_data_dict.get('Height', 0))
    w = float(raw_data_dict.get('Weight', 0))
    abd = float(raw_data_dict.get('Abdomen', 0))
    chest = float(raw_data_dict.get('Chest', 0))
    hip = float(raw_data_dict.get('Hip', 0))

    if h == 0 or w == 0 or hip == 0:
        return 0.0

    wpa = (abd**2) / w
    wthr = abd / h
    whr = abd / hip

    features_list = {
        'Weight': w,
        'Chest': chest,
        'Abdomen': abd,
        'Hip': hip,
        'W_per_A': wpa,
        'WtHR': wthr,
        'WHR': whr
    }

    features_df = pd.DataFrame([features_list])

    print("\n" + " FULL BODYFAT DEBUG V5 ".center(60, "="))
    print(f"User: {raw_data_dict.get('Name', 'Unknown')}")
    print(f"Raw -> H:{h}, W:{w}, Chest:{chest}, Abd:{abd}, Hip:{hip}")

    print("\nDerived:")
    print(f"W_per_A: {wpa:.2f}")
    print(f"WtHR   : {wthr:.4f}")
    print(f"WHR    : {whr:.4f}")

    print("\nModel Input:")
    print(features_df.to_string(index=False))

    try:
        pred = model.predict(features_df)[0]
        print(f"\nPrediction: {pred:.2f}%")
        print("="*60)
        return round(float(pred), 2)
    except Exception as e:
        print("Prediction Error:", e)
        return 0.0