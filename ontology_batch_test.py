import os
import gc
import cv2
import time
import traceback
import numpy as np
import pandas as pd

from core.predictor_v5 import (
    load_model_v5,
    predict_body_fat_v5
)

from core.cv_engine_v5 import (
    process_body_measurements_v5
)

from ontology.ontology_engine import (
    run_ontology
)

# =====================================================
# CONFIG
# =====================================================

FOLDER_PATH = "assets/anh_chuan"

MODEL_PATH = "models/bodyfat_ai_super_clean_v5.pkl"

OUTPUT_FILE = "ontology_evaluation.xlsx"

MAX_FILES = None

# =====================================================
# LOAD MODEL
# =====================================================

print("\nLoading model...")

model_v5 = load_model_v5(MODEL_PATH)

print("Model loaded.")

# =====================================================
# HELPERS
# =====================================================

def safe_join(values):

    if not values:
        return ""

    return ", ".join(map(str, values))


def resize_img(img, max_w=640):

    h, w = img.shape[:2]

    if w > max_w:

        scale = max_w / w

        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale))
        )

    return img


def is_valid_filename(file):

    try:

        parts = file.split("_")

        if len(parts) < 3:
            return False

        nums = parts[2].split(".")[0].split("-")

        return len(nums) >= 6

    except:
        return False


def parse_filename(file_name):

    base = os.path.basename(file_name)

    name = base.split("_")[1]

    data = base.split("_")[2].split(".")[0]

    age, h, w, c, a, hip, *_ = map(
        float,
        data.split("-")
    )

    return {
        "Name": name,
        "Age": age,
        "Height": h,
        "Weight": w,
        "Chest": c,
        "Abdomen_GT": a,
        "Hip_GT": hip
    }

# =====================================================
# GET FILES
# =====================================================

files = sorted([
    f for f in os.listdir(FOLDER_PATH)
    if f.startswith("front")
])

if MAX_FILES:
    files = files[:MAX_FILES]

print("\n" + "=" * 70)
print(" ONTOLOGY BATCH TEST ".center(70, "="))
print("=" * 70)

results = []

# =====================================================
# MAIN LOOP
# =====================================================

for idx, file in enumerate(files):

    print(f"\n[{idx+1}/{len(files)}] {file}")

    try:

        # =================================================
        # VALIDATE FILE
        # =================================================

        if not is_valid_filename(file):

            print(" Invalid filename")
            continue

        info = parse_filename(file)

        front_path = os.path.join(
            FOLDER_PATH,
            file
        )

        side_path = front_path.replace(
            "front",
            "side"
        )

        if not os.path.exists(side_path):

            print(" Missing side image")
            continue

        # =================================================
        # LOAD IMAGES
        # =================================================

        img_f = cv2.imread(front_path)
        img_s = cv2.imread(side_path)

        if img_f is None or img_s is None:

            print(" Cannot read image")
            continue

        img_f = resize_img(img_f)
        img_s = resize_img(img_s)

        # =================================================
        # RAW BASELINE
        # =================================================

        raw_pred = predict_body_fat_v5(
            model_v5,
            info
        )

        # =================================================
        # CV ENGINE
        # =================================================

        scan_result = process_body_measurements_v5(
            img_f,
            img_s,
            info["Height"],
            info["Weight"],
            False
        )

        if scan_result is None:

            print(" Scan result None")
            continue

        if len(scan_result) != 5:

            print(" Invalid scan result")
            continue

        res_scan, _, _, debug_pack, quality_pack = scan_result

        if not res_scan:

            print(" Scan failed")
            continue

        # =================================================
        # AI BODY FAT
        # =================================================

        pred_ai = predict_body_fat_v5(
            model_v5,
            {
                **info,
                **res_scan
            }
        )

        # =================================================
        # QUALITY
        # =================================================

        pose_visibility = quality_pack.get(
            "pose_visibility",
            0.0
        )

        mask_confidence = quality_pack.get(
            "mask_confidence",
            0.0
        )

        # =================================================
        # ONTOLOGY
        # =================================================

        start_time = time.time()

        onto = run_ontology(
            height=info["Height"],
            weight=info["Weight"],

            abdomen=res_scan["Abdomen"],
            hip=res_scan["Hip"],

            predicted_bf=pred_ai,

            pose_visibility=pose_visibility,
            mask_confidence=mask_confidence
        )

        latency = (
            time.time() - start_time
        ) * 1000

        # =================================================
        # EXTRACT OUTPUT
        # =================================================

        explanations = onto.get(
            "Explanations",
            []
        )

        recommendations = onto.get(
            "Recommendations",
            []
        )

        semantic_flags = onto.get(
            "Semantic_Flags",
            []
        )

        triggered_rules = onto.get(
            "Triggered_Rules",
            []
        )

        # =================================================
        # SAVE RESULT
        # =================================================

        results.append({

            # BASIC
            "Name": info["Name"],
            "Age": info["Age"],

            # GT
            "GT_Height": info["Height"],
            "GT_Weight": info["Weight"],
            "GT_Abdomen": info["Abdomen_GT"],
            "GT_Hip": info["Hip_GT"],

            # CV
            "Pred_Abdomen": round(
                res_scan["Abdomen"],
                2
            ),

            "Pred_Hip": round(
                res_scan["Hip"],
                2
            ),

            # BODY FAT
            "BF_Raw": round(raw_pred, 2),
            "BF_AI": round(pred_ai, 2),

            "Delta_BF": round(
                pred_ai - raw_pred,
                2
            ),

            # FEATURES
            "BMI": onto.get("BMI", 0),
            "WHR": onto.get("WHR", 0),
            "WtHR": onto.get("WtHR", 0),

            # SEMANTIC
            "BMI_Class": onto.get(
                "BMI_Class",
                "Unknown"
            ),

            "Fat_Level": onto.get(
                "Fat_Level",
                "Unknown"
            ),

            "Image_Quality": onto.get(
                "Image_Quality",
                "Unknown"
            ),

            # FLAGS
            "Semantic_Flags": safe_join(
                semantic_flags
            ),

            # RULES
            "Triggered_Rules": "\n".join(
                triggered_rules
            ),

            "Rule_Count": len(
                triggered_rules
            ),

            # EXPLANATIONS
            "Explanation_Count": len(
                explanations
            ),

            "Explanations": "\n".join(
                explanations
            ),

            # RECOMMENDATIONS
            "Recommendations": "\n".join(
                recommendations
            ),

            # QUALITY
            "Pose_Visibility": round(
                pose_visibility,
                3
            ),

            "Mask_Confidence": round(
                mask_confidence,
                3
            ),

            # SYSTEM
            "Reasoning_Status": onto.get(
                "Reasoning_Status",
                "Unknown"
            ),

            "Ontology_Latency_ms": round(
                latency,
                2
            )
        })

        # =================================================
        # LOG
        # =================================================

        print(f" BF AI       : {pred_ai:.2f}")

        print(f" BMI         : {onto.get('BMI')}")

        print(f" BMI Class   : {onto.get('BMI_Class')}")

        print(f" Fat Level   : {onto.get('Fat_Level')}")

        print(f" Flags       : {semantic_flags}")

        print(f" Rules       : {len(triggered_rules)}")

    except Exception as e:

        print("\n CRASH DETECTED")
        print(str(e))

        traceback.print_exc()

    finally:

        gc.collect()

# =====================================================
# EXPORT EXCEL
# =====================================================

print("\nExporting Excel...")

df = pd.DataFrame(results)

df.to_excel(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print(" EXPORT SUCCESS ")
print(f" Saved: {OUTPUT_FILE}")
print("=" * 70)