import joblib
import pandas as pd
import streamlit as st
import numpy as np

@st.cache_resource
def load_model(model_path="models/bodyfat_xgboost_model_final.pkl"):
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Lỗi load model: {e}")
        return None

import pandas as pd

def predict_body_fat(model, data_dict):
    # 1. Chuyển dict thành DataFrame
    features = pd.DataFrame([data_dict])
    
    # 2. Đảm bảo ép kiểu float và định dạng .2f cho các cột số để kiểm tra
    # Lưu ý: In ra terminal để soi, không ảnh hưởng đến giá trị thực tế truyền vào model
    debug_features = features.copy()
    for col in debug_features.columns:
        if isinstance(debug_features[col].iloc[0], (int, float, np.float32, np.float64)):
            debug_features[col] = debug_features[col].map('{:.2f}'.format)
    
    # --- DÒNG ÔNG CẦN ĐỂ SOI LỖI ---
    print("\n" + "="*50)
    print(f"DEBUG FEATURES (Tab: {'Heuristic' if data_dict.get('Heuristic') == True else 'Normal'})")
    print(debug_features.to_string(index=False))
    print("="*50)

    # 3. Dự đoán (Phải đảm bảo thứ tự cột khớp với lúc Train Model)
    # Nếu lúc train ông dùng thứ tự khác, hãy reindex lại ở đây
    prediction = model.predict(features)[0]
    
    return round(float(prediction), 2)