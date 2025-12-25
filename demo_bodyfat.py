import joblib
import pandas as pd
import numpy as np

# 1. Tải mô hình đã huấn luyện
try:
    model = joblib.load('tuned_xgboost_k7_final.pkl')
    print("✅ Đã nạp mô hình Tuned XGBoost (k=7) thành công!")
except:
    print("❌ Không tìm thấy file 'tuned_xgboost_k7.pkl'. Hãy đảm bảo file nằm cùng thư mục.")
    exit()

def get_input():
    print("\n" + "="*50)
    print("NHẬP CÁC CHỈ SỐ CƠ THỂ (ĐƠN VỊ: KG & CM)")
    print("="*50)
    
    # Thiên lưu ý: Thứ tự các biến dưới đây ĐƯỢC SẮP XẾP THEO BỘ DỮ LIỆU CHUẨN.
    # Nếu lúc Thiên train có bỏ bớt cột nào, hãy xóa dòng tương ứng ở đây.
    
    try:
        data = {
            'Age': float(input("1. Tuổi (năm): ")),
            'Weight': float(input("2. Cân nặng (kg): ")),
            'Height': float(input("3. Chiều cao (cm): ")),
            'Neck': float(input("4. Vòng cổ (cm): ")),
            'Chest': float(input("5. Vòng ngực (cm): ")),
            'Abdomen': float(input("6. Vòng bụng (cm): ")),
            'Hip': float(input("7. Vòng mông (cm): ")),
            'Thigh': float(input("8. Vòng đùi (cm): ")),
            'Knee': float(input("9. Vòng đầu gối (cm): ")),
            'Ankle': float(input("10. Vòng cổ chân (cm): ")),
            'Biceps': float(input("11. Vòng bắp tay (cm): ")),
            'Forearm': float(input("12. Vòng bắp tay dưới (cm): ")),
            'Wrist': float(input("13. Vòng cổ tay (cm): "))
        }
        return pd.DataFrame([data])
    except ValueError:
        print("❌ Lỗi: Vui lòng chỉ nhập số thực (ví dụ: 75.5)")
        return None

# 2. Chạy chương trình dự đoán
input_df = get_input()

if input_df is not None:
    # Thực hiện dự đoán
    prediction = model.predict(input_df)
    
    print("\n" + "*"*50)
    print(f"KẾT QUẢ DỰ ĐOÁN TỶ LỆ MỠ: {prediction[0]:.2f}%")
    print("*"*50)
    
    # Phân loại nhanh để Thiên có thêm nội dung demo
    res = prediction[0]
    if res < 5: status = "Vận động viên (Rất thấp)"
    elif res < 14: status = "Khỏe mạnh (Vừa phải)"
    elif res < 25: status = "Bình thường"
    else: status = "Cảnh báo: Tỷ lệ mỡ cao"
    
    print(f"Đánh giá sơ bộ: {status}")
    print("="*50)