import os
import cv2
import pandas as pd
from core.cv_engine_new_ver import process_body_measurements
from core.predictor import predict_body_fat, load_model

# Cấu hình
FOLDER_PATH = "assets/anh_chuan"
MODEL_PATH = "models/bodyfat_xgboost_model_final.pkl"

def parse_filename(filename):
    try:
        clean_name = os.path.splitext(filename)[0]
        parts = clean_name.split('_')
        name = parts[1]
        stats = [float(x) for x in parts[2].split('-')]
        return {
            "name": name, "age": int(stats[0]), "h": int(stats[1]), "w": int(stats[2]),
            "r_c": stats[3], "r_a": stats[4], "r_h": stats[5], "r_t": stats[6]
        }
    except: return None

def format_table(title, data):
    df = pd.DataFrame(data)
    print(f"\n" + "="*140)
    print(f"{title:^140}")
    print("="*140)
    with pd.option_context('display.max_columns', None, 'display.width', 1000):
        print(df.to_string(index=False, justify='center'))
    print("="*140)

def run_test():
    model = load_model(MODEL_PATH)
    all_files = sorted(os.listdir(FOLDER_PATH))
    front_images = [f for f in all_files if f.startswith("front_")]
    
    table_1, table_2, table_3 = [], [], []

    for f_file in front_images:
        info = parse_filename(f_file)
        s_file = f_file.replace("front_", "side_")
        if not info or s_file not in all_files: continue

        img_f, img_s = cv2.imread(os.path.join(FOLDER_PATH, f_file)), cv2.imread(os.path.join(FOLDER_PATH, s_file))
        
        # 1. Lấy Ground Truth & BF Real
        data_real = {"Age": info['age'], "Weight": info['w'], "Height": info['h'], "Chest": info['r_c'], "Abdomen": info['r_a'], "Hip": info['r_h'], "Thigh": info['r_t']}
        bf_real = predict_body_fat(model, data_real)

        # 2. Chạy AI Scan (Tab 2)
        res_s, _, _ = process_body_measurements(img_f, img_s, info['h'], info['age'], info['w'], is_loose=False, is_raw=False)
        data_s = {"Age": info['age'], "Weight": info['w'], "Height": info['h'], "Chest": res_s['Chest'], "Abdomen": res_s['Abdomen'], "Hip": res_s['Hip'], "Thigh": res_s['Thigh']}
        bf_s = predict_body_fat(model, data_s)

        # 3. Chạy AI Heuristic (Tab 3)
        res_h, _, _ = process_body_measurements(img_f, img_s, info['h'], info['age'], info['w'], is_loose=True, is_raw=False)
        data_h = {"Age": info['age'], "Weight": info['w'], "Height": info['h'], "Chest": res_h['Chest'], "Abdomen": res_h['Abdomen'], "Hip": res_h['Hip'], "Thigh": res_h['Thigh']}
        bf_h = predict_body_fat(model, data_h)

        # Định dạng cột định danh: Name (Age/H/W)
        id_col = f"{info['name']} ({info['age']}a/{info['h']}cm/{info['w']}kg)"

        # Nạp Bảng 1
        table_1.append({"Mẫu (Tuổi/H/W)": id_col, "Ngực_R": info['r_c'], "Bụng_R": info['r_a'], "Hông_R": info['r_h'], "Đùi_R": info['r_t'], "BF_Real": f"{bf_real}%"})

        # Nạp Bảng 2 (Delta so với Real)
        table_2.append({
            "Mẫu (Tuổi/H/W)": id_col,
            "Δ_Ngực": round(res_s['Chest'] - info['r_c'], 1), "Δ_Bụng": round(res_s['Abdomen'] - info['r_a'], 1),
            "Δ_Hông": round(res_s['Hip'] - info['r_h'], 1), "Δ_Đùi": round(res_s['Thigh'] - info['r_t'], 1),
            "BF_Scan": f"{bf_s}%", "BF_Real": f"{bf_real}%", "Δ_BF": round(abs(bf_s - bf_real), 2)
        })

        # Nạp Bảng 3 (Delta so với Real)
        table_3.append({
            "Mẫu (Tuổi/H/W)": id_col,
            "Δ_Ngực": round(res_h['Chest'] - info['r_c'], 1), "Δ_Bụng": round(res_h['Abdomen'] - info['r_a'], 1),
            "Δ_Hông": round(res_h['Hip'] - info['r_h'], 1), "Δ_Đùi": round(res_h['Thigh'] - info['r_t'], 1),
            "BF_Heu": f"{bf_h}%", "BF_Real": f"{bf_real}%", "Δ_BF": round(abs(bf_h - bf_real), 2)
        })

    format_table("BẢNG 1: THÔNG SỐ ĐO THỰC TẾ & BODYFAT CHUẨN (GROUND TRUTH)", table_1)
    format_table("BẢNG 2: SAI SỐ CHI TIẾT TAB AI SCAN (DỰA TRÊN HỆ SỐ BMI)", table_2)
    format_table("BẢNG 3: SAI SỐ CHI TIẾT TAB AI HEURISTIC (CHẾ ĐỘ GIẢM NHIỄU TRANG PHỤC)", table_3)

if __name__ == "__main__":
    run_test()