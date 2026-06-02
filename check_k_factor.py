import os

import cv2
import mediapipe as mp


# Sample configuration
# Add new front-view images and their real height values here.
SAMPLES = {
    "front_D_new.jpg": 167.0,
    "front_H_new.jpg": 158.0,
    "front_L_new.jpg": 169.0,
    "front_T_new.jpg": 163.0,
    "front_K_new.jpg": 165.0,
}

ASSETS_FOLDER = "assets"
K_RANGE = [2.3, 2.4, 2.5, 2.6, 2.7]

mp_pose = mp.solutions.pose


def calculate_height_ratio(image, real_height, k_values):
    """
    Calculate height scaling ratios using different head compensation factors.
    """

    image_height, _, _ = image.shape

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark

    y_nose = landmarks[mp_pose.PoseLandmark.NOSE].y * image_height
    y_eye = (
        (
            landmarks[mp_pose.PoseLandmark.LEFT_EYE].y
            + landmarks[mp_pose.PoseLandmark.RIGHT_EYE].y
        )
        / 2
    ) * image_height
    y_heel = (
        (
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y
            + landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y
        )
        / 2
    ) * image_height

    eye_to_nose_distance = abs(y_nose - y_eye)
    nose_to_heel_distance = abs(y_heel - y_nose)

    ratios = []

    for k_value in k_values:
        total_pixel_height = nose_to_heel_distance + (eye_to_nose_distance * k_value)

        if total_pixel_height == 0:
            ratios.append(None)
            continue

        ratios.append(real_height / total_pixel_height)

    return ratios


def run_test():
    """
    Test multiple head compensation factors and print the scaling ratios.
    """

    header = (
        f"{'IMAGE':<15} | "
        f"{'REAL H':<8} | "
        f"{'k=2.3':<8} | "
        f"{'k=2.4':<8} | "
        f"{'k=2.5':<8} | "
        f"{'k=2.6':<8} | "
        f"{'k=2.7':<8}"
    )

    print(header)
    print("-" * 80)

    for image_name, real_height in SAMPLES.items():
        image_path = os.path.join(ASSETS_FOLDER, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"{image_name:<15} | File not found in folder: {ASSETS_FOLDER}")
            continue

        ratios = calculate_height_ratio(image, real_height, K_RANGE)

        if ratios is None:
            print(f"{image_name:<15} | Pose landmarks not detected")
            continue

        row = f"{image_name:<15} | {real_height:<8} |"

        for ratio in ratios:
            ratio_text = f"{ratio:.4f}" if ratio is not None else "N/A"
            row += f" {ratio_text:<8} |"

        print(row)


if __name__ == "__main__":
    run_test()