import cv2
import os
import numpy as np

from cv_engine_v5 import process_body_measurements_v5  # sửa đúng tên file bạn đang dùng

# ==============================
# 🔧 CONFIG TEST
# ==============================

# 👉 Bạn chỉnh các config ở đây
Y_MAP_CONFIGS = [
    {"name": "A_base", "Chest": 0.25, "Abdomen": 0.25, "Hip": 0.00},
    {"name": "B_current", "Chest": 0.27, "Abdomen": 0.30, "Hip": 0.05},
    {"name": "C_high", "Chest": 0.30, "Abdomen": 0.35, "Hip": 0.08},
    {"name": "D_low", "Chest": 0.20, "Abdomen": 0.20, "Hip": 0.00},
]

IMG_DIR = "assets/anh_chuan"

# 👉 giả lập chiều cao / cân nặng (cho ổn định)
REAL_HEIGHT = 170
WEIGHT = 65


# ==============================
# 🔁 PATCH y_map vào engine
# ==============================

def process_with_custom_y_map(front_img, side_img, real_h, weight, y_map_cfg):
    """
    Override y_map trong hàm gốc
    """

    import cv_engine_v5 as engine

    # backup hàm gốc
    original_func = engine.process_body_measurements_v5

    def patched(front_img, side_img, real_h, weight, use_long_pants=False):
        mask_f, mask_raw_f, res_f = engine.get_body_data_v5(front_img)
        mask_s, mask_raw_s, res_s = engine.get_body_data_v5(side_img)

        if not all([res_f, res_f.pose_landmarks, res_s, res_s.pose_landmarks]):
            return None, None, None, None

        h_img, w_img, _ = front_img.shape
        lm_f = res_f.pose_landmarks.landmark
        lm_s = res_s.pose_landmarks.landmark

        # ===== SCALE =====
        y_nose = lm_f[0].y * h_img
        y_heel = ((lm_f[29].y + lm_f[30].y) / 2) * h_img
        head_offset = abs(y_nose - (lm_f[1].y * h_img)) * 2.5
        ratio = real_h / abs(y_heel - (y_nose - head_offset))

        # ===== BMI =====
        bmi = weight / ((real_h / 100) ** 2)
        f_calib = 1.12 if bmi < 18.5 else (1.204 if bmi < 25 else 1.25)

        shoulder = lm_f[11].y
        hip = lm_f[23].y
        torso = hip - shoulder

        # 🔥 DÙNG CONFIG
        y_map_front = {
            'Chest': shoulder + torso * y_map_cfg['Chest'],
            'Abdomen': hip - torso * y_map_cfg['Abdomen'],
            'Hip': hip + torso * y_map_cfg['Hip']
        }

        results = {}

        for part in ['Chest', 'Abdomen', 'Hip']:

            y_f = y_map_front[part]
            iterator = engine.get_iterator(bmi, part)

            w_v, _, _ = engine.get_dimension_at_y_v5(
                mask_f, y_f, lm_f, part, ratio, iterator
            )

            d_v, _, _, _ = engine.find_best_depth(
                mask_s, y_f, lm_s, part, ratio, iterator
            )

            if w_v == 0 or d_v == 0:
                continue

            a, b = w_v / 2, d_v / 2
            h_el = ((a - b)**2) / ((a + b)**2) if (a + b) != 0 else 0

            circum = np.pi * (a + b) * (
                1 + (3 * h_el) / (10 + np.sqrt(4 - 3 * h_el))
            ) if (a + b) != 0 else 0

            results[part] = circum * f_calib

        return results

    # chạy
    result = patched(front_img, side_img, real_h, weight)

    return result


# ==============================
# 📊 RUN TEST
# ==============================

def run_test():
    images = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg") or f.endswith(".png")]

    print("\n" + "="*80)
    print(" TEST Y_MAP CONFIGS ".center(80, "="))

    for img_name in images:

        img_path = os.path.join(IMG_DIR, img_name)

        front = cv2.imread(img_path)
        side = front.copy()  # 👉 tạm dùng front làm side nếu bạn chưa có

        print(f"\n🖼 IMAGE: {img_name}")

        results_table = []

        for cfg in Y_MAP_CONFIGS:

            res = process_with_custom_y_map(
                front, side,
                REAL_HEIGHT,
                WEIGHT,
                cfg
            )

            if res is None:
                continue

            chest = res.get("Chest", 0)
            abdomen = res.get("Abdomen", 0)
            hip = res.get("Hip", 0)

            score = chest + abdomen + hip  # 👉 metric đơn giản

            results_table.append({
                "name": cfg["name"],
                "Chest": chest,
                "Abdomen": abdomen,
                "Hip": hip,
                "Score": score
            })

        # ===== SORT =====
        results_table = sorted(results_table, key=lambda x: x["Score"], reverse=True)

        # ===== PRINT =====
        print("\nCONFIG COMPARISON:")
        print(f"{'Name':<12} | {'Chest':<8} | {'Abdomen':<10} | {'Hip':<8} | {'Score'}")
        print("-"*60)

        for r in results_table:
            print(f"{r['name']:<12} | {r['Chest']:<8.1f} | {r['Abdomen']:<10.1f} | {r['Hip']:<8.1f} | {r['Score']:.1f}")

        best = results_table[0]["name"]
        print(f"\n👉 BEST CONFIG: {best}")

        # show image
        cv2.imshow("Test Image", front)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_test()