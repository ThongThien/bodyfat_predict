import pandas as pd
import numpy as np

def create_final_dts(input_path, output_path):
    print("--- Bắt đầu xử lý thêm Feature W_per_A ---")
    
    # 1. Đọc dữ liệu từ file bạn đã chuẩn hóa (như trong hình)
    try:
        df = pd.read_csv(input_path)
    except:
        # Nếu là file Excel thì dùng read_excel
        df = pd.read_excel(input_path)
        
    print(f"Số lượng dòng nạp vào: {len(df)}")

    # 2. Tính toán biến W_per_A (Khuếch đại tương tác Bụng và Cân nặng)
    # Công thức: Abdomen bình phương chia cho Weight
    # Giúp Model phân biệt rõ người có bụng to thực sự so với người có khung xương nặng
    df['W_per_A'] = (df['Abdomen']**2) / df['Weight']

    # 3. Kiểm tra và đảm bảo các biến phái sinh khác đã có sẵn hoặc tính lại cho chuẩn
    # (Đề phòng trường hợp file input chưa có hoặc tính sai)
    df['WHR'] = df['Abdomen'] / df['Hip']
    df['WtHR'] = df['Abdomen'] / df['Height']

    # 4. Sắp xếp lại danh sách cột theo đúng yêu cầu Input Model
    # Thứ tự này rất quan trọng để khi Predict không bị lệch cột
    final_cols = [
        'BodyFat',   # Biến mục tiêu (Target)
        'Weight',    # Feature 1
        'Chest',     # Feature 2
        'Abdomen',   # Feature 3
        'Hip',       # Feature 4
        'W_per_A',   # Feature 5 (Biến mới)
        'WtHR',      # Feature 6
        'WHR',       # Feature 7
        'Height',    # Giữ lại để tham khảo (Model có thể bỏ qua)
        'Age'        # Giữ lại nếu có trong dts gốc
    ]

    # Kiểm tra xem Age có tồn tại không, nếu không thì bỏ ra khỏi list
    final_cols = [col for col in final_cols if col in df.columns]
    
    dts_final = df[final_cols]

    # 5. Xem qua kết quả 5 dòng đầu
    print("\n[LOG] 5 dòng đầu của Dataset mới:")
    print(dts_final[['BodyFat', 'Weight', 'Abdomen', 'W_per_A']].head())

    # 6. Xuất file CSV cuối cùng
    dts_final.to_csv(output_path, index=False)
    
    print("-" * 30)
    print(f"[SUCCESS] Đã tạo thành công file: {output_path}")
    print(f"Tổng số dòng: {len(dts_final)}")
    print(f"Danh sách Feature: {dts_final.columns.tolist()}")

if __name__ == "__main__":
    # Bạn hãy thay đổi tên file input đúng với file của bạn (csv hoặc xlsx)
    create_final_dts('bodyfat_final_v1.csv', 'bodyfat_final_v2.csv')