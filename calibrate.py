import os
import cv2
import numpy as np
import pandas as pd
from core.cv_engine_new_ver import process_body_measurements

FOLDER_PATH = "assets/anh_chuan"
OUTPUT_REPORT = "calibration_detailed_report.csv"

def parse_filename(filename):
    try:
        clean_name = os.path.splitext(filename)[0]
        parts = clean_name.split('_')
        name = parts[1]
        stats = [float(x) for x in parts[2].split('-')]
        return {"name": name, "age": int(stats[0]), "h": stats[1], "w": stats[2],
                "real_c": stats[3], "real_a": stats[4], "real_h": stats[5], "real_t": stats[6]}
    except: return None

def run_calibration():
    if not os.path.exists(FOLDER_PATH): return
    all_files = os.listdir(FOLDER_PATH)
    front_images = [f for f in all_files if f.startswith("front_")]
    data_list = []

    for f_file in front_images:
        s_file = f_file.replace("front_", "side_")
        if s_file not in all_files: continue
        info = parse_filename(f_file)
        if not info: continue

        img_f, img_s = cv2.imread(os.path.join(FOLDER_PATH, f_file)), cv2.imread(os.path.join(FOLDER_PATH, s_file))
        
        # Gọi engine mới (trả về 4 tham số)
        raw_res, _, _, raw_dims = process_body_measurements(img_f, img_s, info['h'], info['age'], info['w'], is_raw=True)

        if raw_res:
            row = {"Name": info['name'], "BMI": round(info['w']/((info['h']/100)**2), 2), 
                   "Ratio_F": round(raw_dims['Abdomen']['ratio_f'], 4),
                   "Ratio_S": round(raw_dims['Abdomen']['ratio_s'], 4)}
            
            for part, p_short in [('Chest', 'C'), ('Abdomen', 'A'), ('Hip', 'H'), ('Thigh', 'T')]:
                d = raw_dims[part]
                row.update({
                    f"Real_{p_short}": info[f'real_{p_short.lower()}'],
                    f"Raw_{p_short}": raw_res[part],
                    f"W_px_{part}": d['w_px'],
                    f"D_px_{part}": d['d_px'],
                    f"W_cm_{part}": round(d['w_cm'], 2),
                    f"D_cm_{part}": round(d['d_cm'], 2),
                    f"f_{part}": round(info[f'real_{p_short.lower()}'] / raw_res[part], 3)
                })
            
            row["f_Avg_Person"] = round((row['f_Chest'] + row['f_Abdomen'] + row['f_Hip'] + row['f_Thigh'])/4, 3)
            data_list.append(row)

    df = pd.DataFrame(data_list)
    df.to_csv(OUTPUT_REPORT, index=False)
    print(f"📊 Báo cáo chi tiết đã xuất: {OUTPUT_REPORT}")
    print(f"Hệ số f_system trung bình: {round(df['f_Avg_Person'].mean(), 4)}")

if __name__ == "__main__":
    run_calibration()