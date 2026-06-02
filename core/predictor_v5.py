import os

import joblib
import pandas as pd
import streamlit as st


@st.cache_resource
def load_model_v5(model_path: str = "models/bodyfat_ai_super_clean_v5.pkl"):
    """
    Load the trained Body Fat prediction model.
    """

    if not os.path.exists(model_path):
        st.error(f"Model not found: {model_path}")
        return None

    return joblib.load(model_path)


def predict_body_fat_v5(model, raw_data_dict):
    """
    Predict body fat percentage using engineered body measurements.
    """

    height = float(raw_data_dict.get("Height", 0))
    weight = float(raw_data_dict.get("Weight", 0))
    abdomen = float(raw_data_dict.get("Abdomen", 0))
    chest = float(raw_data_dict.get("Chest", 0))
    hip = float(raw_data_dict.get("Hip", 0))

    # Prevent division-by-zero errors
    if height == 0 or weight == 0 or hip == 0:
        return 0.0

    # Derived features
    w_per_a = (abdomen**2) / weight
    wthr = abdomen / height
    whr = abdomen / hip

    features = {
        "Weight": weight,
        "Chest": chest,
        "Abdomen": abdomen,
        "Hip": hip,
        "W_per_A": w_per_a,
        "WtHR": wthr,
        "WHR": whr,
    }

    features_df = pd.DataFrame([features])

    # Debug information
    print("\n" + " BODY FAT MODEL V5 DEBUG ".center(60, "="))
    print(f"User: {raw_data_dict.get('Name', 'Unknown')}")
    print(
        f"Raw Measurements -> "
        f"Height: {height}, Weight: {weight}, "
        f"Chest: {chest}, Abdomen: {abdomen}, Hip: {hip}"
    )

    print("\nDerived Features:")
    print(f"W_per_A: {w_per_a:.2f}")
    print(f"WtHR   : {wthr:.4f}")
    print(f"WHR    : {whr:.4f}")

    print("\nModel Input:")
    print(features_df.to_string(index=False))

    try:
        prediction = model.predict(features_df)[0]

        print(f"\nPredicted Body Fat: {prediction:.2f}%")
        print("=" * 60)

        return round(float(prediction), 2)

    except Exception as error:
        print(f"Prediction Error: {error}")
        return 0.0