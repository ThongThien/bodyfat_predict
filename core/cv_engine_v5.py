import numpy as np
import cv2
import mediapipe as mp
import os

os.environ["OMP_NUM_THREADS"] = "1"  # tránh crash thread

mp_pose = mp.solutions.pose
mp_segmentation = mp.solutions.selfie_segmentation

POSE_MODEL = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=1,
    enable_segmentation=True,
    min_detection_confidence=0.5
)

SEG_MODEL = mp_segmentation.SelfieSegmentation(model_selection=1)
POSE_MODEL = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def safe_resize(img, max_size=1024):
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

def get_body_data_v5(img, debug=False):
    try:
        if img is None:
            print(" Image None")
            return None, None, None

        # Resize để tránh crash
        img = safe_resize(img)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #  TRY CATCH MEDIAPIPE
        try:
            res_pose = POSE_MODEL.process(img_rgb)
        except Exception as e:
            print(" Mediapipe crash:", e)
            return None, None, None

        if not res_pose or not res_pose.pose_landmarks:
            print(" No landmarks detected")
            return None, None, None

        # ===== SEGMENTATION =====
        mask = None
        if res_pose.segmentation_mask is not None:
            mask = (res_pose.segmentation_mask > 0.5).astype("uint8") * 255

        # ===== DEBUG =====
        if debug:
            return mask, mask, res_pose

        return mask, None, res_pose

    except Exception as e:
        print(" get_body_data_v5 error:", e)
        return None, None, None

def get_dimension_at_y_v5(mask, y_norm, lm_list, part_name, ratio, iterator=1):
    h_img, w_img = mask.shape
    y_pixel = min(int(y_norm * h_img), h_img - 1)

    shoulder_w = abs(lm_list[12].x - lm_list[11].x) * w_img

    base_margin = 0.2
    if part_name == 'Hip':
        base_margin = 0.25

    margin = base_margin * iterator

    x_min = max(0, int(min(lm_list[11].x, lm_list[12].x) * w_img - shoulder_w * margin))
    x_max = min(w_img - 1, int(max(lm_list[11].x, lm_list[12].x) * w_img + shoulder_w * margin))

    row = mask[y_pixel, x_min:x_max]
    white_pixels = np.where(row)[0]

    if len(white_pixels) < 2:
        return 0, 0, 0

    x1, x2 = np.min(white_pixels), np.max(white_pixels)
    width_cm = (x2 - x1) * ratio

    return width_cm, x_min + x1, x_min + x2


# 🔥 NEW: scan depth tốt nhất (lấy từ bản 2)
def find_best_depth(mask, y_center, lm, part, ratio, iterator):
    offsets = [-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03]

    best_d = 0
    best_y = y_center
    best_x1, best_x2 = 0, 0

    for off in offsets:
        y_try = y_center + off

        d_v, x1, x2 = get_dimension_at_y_v5(
            mask, y_try, lm, part, ratio, iterator
        )

        if d_v > best_d:
            best_d = d_v
            best_y = y_try
            best_x1, best_x2 = x1, x2

    return best_d, best_y, best_x1, best_x2


def get_iterator(bmi, part, use_long_pants=False):
    if part == 'Hip':
        base = 1.6 if not use_long_pants else 1.7
    else:
        base = 1.0

    scale = 1 + (bmi - 22) * 0.02

    return np.clip(base * scale, 0.8, 2.2)


def process_body_measurements_v5(front_img, side_img, real_h, weight, use_long_pants=False):
    try:
        # ===== VALIDATE INPUT =====
        if front_img is None or side_img is None:
            print("❌ Input image None")
            return None, None, None, None

        # ===== SAFE RESIZE (tránh crash mediapipe) =====
        def safe_resize(img, max_size=1024):
            if img is None:
                return None
            h, w = img.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            return img

        front_img = safe_resize(front_img)
        side_img = safe_resize(side_img)

        # ===== GET BODY DATA (TRY SAFE) =====
        try:
            mask_f, mask_raw_f, res_f = get_body_data_v5(front_img, debug=True)
            mask_s, mask_raw_s, res_s = get_body_data_v5(side_img, debug=True)
        except Exception as e:
            print("❌ Mediapipe processing crash:", e)
            return None, None, None, None

        # ===== VALIDATE LANDMARK =====
        if not all([
            res_f, hasattr(res_f, "pose_landmarks"), res_f.pose_landmarks,
            res_s, hasattr(res_s, "pose_landmarks"), res_s.pose_landmarks
        ]):
            print("❌ Không detect được pose")
            return None, None, None, None

        try:
            h_img, w_img, _ = front_img.shape
            lm_f = res_f.pose_landmarks.landmark
            lm_s = res_s.pose_landmarks.landmark
        except Exception as e:
            print("❌ Landmark error:", e)
            return None, None, None, None

        # ===== SCALE =====
        try:
            y_nose = lm_f[0].y * h_img
            y_heel = ((lm_f[29].y + lm_f[30].y) / 2) * h_img
            head_offset = abs(y_nose - (lm_f[1].y * h_img)) * 2.5

            denom = abs(y_heel - (y_nose - head_offset))
            if denom == 0:
                print("❌ Scale division by 0")
                return None, None, None, None

            ratio = real_h / denom
        except Exception as e:
            print("❌ Scale calc error:", e)
            return None, None, None, None

        # ===== BMI CALIB =====
        try:
            bmi = weight / ((real_h / 100) ** 2)
            f_calib = 1.12 if bmi < 18.5 else (1.204 if bmi < 25 else 1.25)
        except:
            bmi = 22
            f_calib = 1.2

        # ===== BASE LANDMARK =====
        try:
            shoulder = lm_f[11].y
            hip = lm_f[23].y
            torso = hip - shoulder
        except Exception as e:
            print("❌ Torso calc error:", e)
            return None, None, None, None

        y_map_front = {
            'Chest': shoulder + torso * 0.27,
            'Abdomen': hip - torso * 0.30,
            'Hip': hip + torso * 0.05
        }

        results = {}
        viz_f, viz_s = front_img.copy(), side_img.copy()

        print("\n" + "="*60)
        print(" FULL DEBUG MEASUREMENTS ".center(60, "="))

        for part in ['Chest', 'Abdomen', 'Hip']:
            try:
                y_f = y_map_front[part]

                # ===== ITERATOR =====
                iterator = get_iterator(bmi, part, use_long_pants)

                # ===== WIDTH =====
                w_v, x1f, x2f = get_dimension_at_y_v5(
                    mask_f, y_f, lm_f, part, ratio, iterator
                )

                # ===== DEPTH =====
                d_v, y_s, x1s, x2s = find_best_depth(
                    mask_s, y_f, lm_s, part, ratio, iterator
                )

                # ===== HANDLE MISSING =====
                if w_v == 0 or d_v == 0:
                    print(f"[WARN] Missing data at {part}")
                    continue

                # ===== ELLIPSE =====
                a, b = w_v / 2, d_v / 2

                if (a + b) == 0:
                    continue

                h_el = ((a - b)**2) / ((a + b)**2)

                circum_raw = np.pi * (a + b) * (
                    1 + (3 * h_el) / (10 + np.sqrt(4 - 3 * h_el))
                )

                circum_final = round(circum_raw * f_calib, 2)

                if use_long_pants and part == 'Hip':
                    circum_final *= 0.9  

                results[part] = circum_final

                # ===== LOG =====
                print(f"\n--- {part.upper()} ---")
                print(f"y_front: {y_f:.4f} | y_side(best): {y_s:.4f}")
                print(f"Width: {w_v:.2f} cm")
                print(f"Depth: {d_v:.2f} cm")
                print(f"Circum: {circum_final:.2f}")

                # ===== DRAW SAFE =====
                try:
                    y_px_f = int(y_f * h_img)
                    y_px_s = int(y_s * h_img)

                    cv2.line(viz_f, (int(x1f), y_px_f), (int(x2f), y_px_f), (0, 255, 0), 3)
                    cv2.line(viz_s, (int(x1s), y_px_s), (int(x2s), y_px_s), (0, 255, 0), 3)

                    cv2.circle(viz_f, (int(w_img/2), y_px_f), 5, (0, 0, 255), -1)
                    cv2.circle(viz_s, (int(w_img/2), y_px_s), 5, (0, 0, 255), -1)
                except:
                    pass

            except Exception as e:
                print(f"❌ Error at {part}:", e)
                continue

        print("="*60)

        # ===== FINAL CHECK =====
        if len(results) == 0:
            print("❌ No valid measurements")
            return None, None, None, None

        debug_pack = {
            "mask_f": mask_f,
            "mask_s": mask_s,
            "mask_raw_f": mask_raw_f,
            "mask_raw_s": mask_raw_s
        }

        return results, viz_f, viz_s, debug_pack

    except Exception as e:
        print("❌ FATAL ERROR:", e)
        return None, None, None, None