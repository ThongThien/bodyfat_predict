import os
import gc
import cv2
import time
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

MAX_FILES = 20

# =====================================================
# LOAD MODEL
# =====================================================

model_v5 = load_model_v5(MODEL_PATH)

# =====================================================
# HELPERS
# =====================================================

def is_valid_filename(file):

    try:
        parts = file.split("_")

        if len(parts) < 3:
            return False

        data = parts[2].split(".")[0]

        nums = data.split("-")

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
        "Abdomen": a,
        "Hip": hip
    }


def resize_img(img, max_w=640):

    h, w = img.shape[:2]

    if w > max_w:

        scale = max_w / w

        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale))
        )

    return img


# =====================================================
# MAIN TEST
# =====================================================

results = []

files = [
    f for f in os.listdir(FOLDER_PATH)
    if f.startswith("front")
]

files = files[:MAX_FILES]

print("\n" + "=" * 60)
print(" ONTOLOGY BATCH TEST ".center(60, "="))
print("=" * 60)

for idx, file in enumerate(files):

    print(f"\n[{idx+1}/{len(files)}] {file}")

    if not is_valid_filename(file):

        print(" Invalid filename")
        continue

    try:

        info = parse_filename(file)

    except Exception as e:

        print(" Parse error:", e)
        continue

    path_f = os.path.join(FOLDER_PATH, file)

    path_s = path_f.replace("front", "side")

    if not os.path.exists(path_s):

        print(" Missing side image")
        continue

    try:

        # =================================================
        # LOAD IMAGE
        # =================================================

        img_f = cv2.imread(path_f)
        img_s = cv2.imread(path_s)

        if img_f is None or img_s is None:

            print(" Cannot read image")
            continue

        img_f = resize_img(img_f)
        img_s = resize_img(img_s)

        # =================================================
        # RAW BF
        # =================================================

        raw_pred = predict_body_fat_v5(
            model_v5,
            info
        )

        # =================================================
        # AI SCAN
        # =================================================

        res_scan, _, _, _, quality_pack = (
            process_body_measurements_v5(
                img_f,
                img_s,
                info["Height"],
                info["Weight"],
                False
            )
        )

        if not res_scan:

            print(" Scan failed")
            continue

        # =================================================
        # AI BF
        # =================================================

        pred_ai = predict_body_fat_v5(
            model_v5,
            {
                **info,
                **res_scan
            }
        )

        # =================================================
        # ONTOLOGY
        # =================================================

        start_time = time.time()

        onto = run_ontology(
            height=info["Height"],
            weight=info["Weight"],
            chest=res_scan["Chest"],
            abdomen=res_scan["Abdomen"],
            hip=res_scan["Hip"],
            predicted_bf=pred_ai,

            pose_visibility=quality_pack["pose_visibility"],
            mask_confidence=quality_pack["mask_confidence"],
            missing_landmarks=quality_pack["missing_landmarks"]
        )

        latency = time.time() - start_time

        # =================================================
        # RULE ACTIVATION
        # =================================================

        rule_count = (
            len(onto["explanations"])
            + len(onto["fat_distribution"])
            + len(onto["semantic_flags"])
        )

        # =================================================
        # SAVE RESULT
        # =================================================

        results.append({

            # BASIC
            "Name": info["Name"],

            # GROUND TRUTH
            "BF_Raw": round(raw_pred, 2),

            # AI RESULT
            "BF_AI": round(pred_ai, 2),

            "Delta_BF": round(
                pred_ai - raw_pred,
                2
            ),

            # BODY INDEX
            "BMI": onto["bmi"],
            "WHR": onto["whr"],
            "WtHR": onto["wthr"],

            # REASONING
            "BMI_Class": onto["bmi_class"],
            "Fat_Level": onto["fat_level"],
            "Quality": onto["quality"],

            # FLAGS
            "Semantic_Flags": ", ".join(
                onto["semantic_flags"]
            ),

            "Fat_Distribution": ", ".join(
                onto["fat_distribution"]
            ),

            # EXPLANATION
            "Explanation_Count": len(
                onto["explanations"]
            ),

            "Rule_Activation_Count": rule_count,

            "Explanations": "\n".join(
                onto["explanations"]
            ),

            # RECOMMENDATION
            "Recommendations": "\n".join(
                onto["recommendations"]
            ),

            # LATENCY
            "Ontology_Latency (ms)": round(
                latency*1000,
                2
            )
        })

        print(f" BF Raw: {raw_pred:.2f}")
        print(f" BF AI : {pred_ai:.2f}")
        print(f" BMI   : {onto['bmi']}")
        print(f" Rules : {rule_count}")

    except Exception as e:

        print(" Crash:", e)

    finally:

        gc.collect()

# =====================================================
# EXPORT EXCEL
# =====================================================

df = pd.DataFrame(results)

df.to_excel(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print(" EXPORT SUCCESS ")
print(f" Saved: {OUTPUT_FILE}")
print("=" * 60)