import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Initialize MediaPipe models.
mp_pose = mp.solutions.pose
mp_segmentation = mp.solutions.selfie_segmentation


def get_body_data_research(img_bgr):
    """Extract segmentation mask and pose landmarks from an image."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with mp_segmentation.SelfieSegmentation(model_selection=1) as seg, \
            mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        seg_result = seg.process(img_rgb)
        pose_result = pose.process(img_rgb)

        mask = seg_result.segmentation_mask > 0.5

    return mask, pose_result


def refine_body_mask_research(mask_raw, landmarks, img_h, lower_iterations):
    """Refine body mask by applying different erosion levels to upper and lower body."""
    hip_y_px = int(((landmarks[23].y + landmarks[24].y) / 2) * img_h)
    kernel = np.ones((3, 3), np.uint8)

    upper_mask = mask_raw.copy().astype(np.uint8)
    upper_mask[hip_y_px:, :] = 0
    upper_mask_refined = cv2.erode(upper_mask, kernel, iterations=1)

    lower_mask = mask_raw.copy().astype(np.uint8)
    lower_mask[:hip_y_px, :] = 0
    lower_mask_refined = cv2.erode(lower_mask, kernel, iterations=lower_iterations)

    return cv2.bitwise_or(upper_mask_refined, lower_mask_refined)


def calculate_ramanujan(width_cm, depth_cm):
    """Estimate ellipse circumference using Ramanujan approximation."""
    a = width_cm / 2
    b = depth_cm / 2

    if a <= 0 or b <= 0:
        return 0

    h_el = ((a - b) ** 2) / ((a + b) ** 2)

    return np.pi * (a + b) * (
        1 + (3 * h_el) / (10 + np.sqrt(4 - 3 * h_el))
    )


SAMPLES = {
    "D": {"front": "front_D_new.jpg", "side": "side_D_new.jpg", "height": 167.0, "real_hip": 93.0},
    "H": {"front": "front_H_new.jpg", "side": "side_H_new.jpg", "height": 158.0, "real_hip": 86.0},
    "L": {"front": "front_L_new.jpg", "side": "side_L_new.jpg", "height": 169.0, "real_hip": 90.0},
    "T": {"front": "front_T_new.jpg", "side": "side_T_new.jpg", "height": 163.0, "real_hip": 86.0},
    "K": {"front": "front_K_new.jpg", "side": "side_K_new.jpg", "height": 165.0, "real_hip": 93.0},
}

K_FACTOR = 2.5
ITERATION_LIST = [0, 2, 4, 6]

results_table = []

print("Processing research samples with the CV model...")

for sample_name, sample_data in SAMPLES.items():
    front_img = cv2.imread(f"assets/{sample_data['front']}")
    side_img = cv2.imread(f"assets/{sample_data['side']}")

    if front_img is None or side_img is None:
        continue

    img_h, _, _ = front_img.shape

    front_mask_raw, front_pose = get_body_data_research(front_img)
    side_mask_raw, side_pose = get_body_data_research(side_img)

    if not front_pose.pose_landmarks or not side_pose.pose_landmarks:
        continue

    front_landmarks = front_pose.pose_landmarks.landmark
    side_landmarks = side_pose.pose_landmarks.landmark

    # Calculate pixel-to-centimeter scale using estimated body height.
    nose_y = front_landmarks[0].y * img_h
    heel_y = ((front_landmarks[29].y + front_landmarks[30].y) / 2) * img_h
    head_top_offset = abs(nose_y - (front_landmarks[1].y * img_h)) * K_FACTOR

    ratio = sample_data["height"] / abs(heel_y - (nose_y - head_top_offset))

    # Define hip scanning position.
    hip_y_norm = front_landmarks[23].y + 0.08
    hip_y_px = int(hip_y_norm * img_h)

    row_data = {
        "Sample": sample_name,
        "Real Hip": sample_data["real_hip"],
    }

    for iteration in ITERATION_LIST:
        refined_front_mask = refine_body_mask_research(
            front_mask_raw,
            front_landmarks,
            img_h,
            iteration,
        )

        refined_side_mask = refine_body_mask_research(
            side_mask_raw,
            side_landmarks,
            img_h,
            iteration,
        )

        # Count body pixels at the selected hip Y position.
        front_width_px = np.sum(refined_front_mask[hip_y_px, :] > 0)
        side_depth_px = np.sum(refined_side_mask[hip_y_px, :] > 0)

        calculated_hip = calculate_ramanujan(
            front_width_px * ratio,
            side_depth_px * ratio,
        )

        row_data[f"Iter_{iteration}"] = round(calculated_hip, 2)

    results_table.append(row_data)

results_df = pd.DataFrame(results_table)

print("\nHIP CIRCUMFERENCE COMPARISON BY MASK REFINEMENT ITERATION")
print(results_df.to_string(index=False))