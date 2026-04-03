import cv2
import mediapipe as mp
import os

# --- CẤU HÌNH QUY MÔ (SCALE) ---
# Sau này có thêm mẫu, ông chỉ cần quăng ảnh vào folder 'asset' 
# và thêm dòng vào danh sách dưới đây:
samples = {
    "front_D_new.jpg": 167.0,
    "front_H_new.jpg": 158.0,
    "front_L_new.jpg": 169.0,
    "front_T_new.jpg": 163.0,
    "front_K_new.jpg": 165.0,
    # "file_moi.png": 172.0,
}

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def run_test():
    folder = 'assets'
    k_range = [2.3, 2.4, 2.5, 2.6, 2.7]
    
    print(f"{'IMAGE':<15} | {'REAL H':<8} | {'k=2.3':<8} | {'k=2.4':<8} | {'k=2.5':<8} | {'k=2.6':<8} | {'k=2.7':<8}")
    print("-" * 80)

    for img_name, real_h in samples.items():
        img_path = os.path.join(folder, img_name)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"❌ Lỗi: Không tìm thấy file {img_name} trong thư mục {folder}")
            continue
            
        h_img, _, _ = image.shape
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks:
            print(f"⚠️ {img_name:<13} | Không bắt được Pose")
            continue
            
        lm = results.pose_landmarks.landmark
        
        # 1. Lấy tọa độ Y (pixel)
        y_nose = lm[mp_pose.PoseLandmark.NOSE].y * h_img
        y_eye = ((lm[mp_pose.PoseLandmark.LEFT_EYE].y + lm[mp_pose.PoseLandmark.RIGHT_EYE].y) / 2) * h_img
        y_heel = ((lm[mp_pose.PoseLandmark.LEFT_ANKLE].y + lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y) / 2) * h_img
        
        # 2. Tính các khoảng cách cơ bản
        dist_eye_nose = abs(y_nose - y_eye)
        dist_nose_heel = abs(y_heel - y_nose)

        # 3. Tính Ratio cho từng k và in trực tiếp ra dòng
        row_str = f"{img_name:<15} | {real_h:<8} |"
        for k in k_range:
            p_total = dist_nose_heel + (dist_eye_nose * k)
            ratio = real_h / p_total
            row_str += f" {ratio:.4f} |"
        
        print(row_str)

if __name__ == "__main__":
    run_test()