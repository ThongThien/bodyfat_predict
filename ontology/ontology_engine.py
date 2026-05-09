from owlready2 import *
import uuid
import numpy as np


def run_ontology(
    height,
    weight,
    chest,
    abdomen,
    hip,
    predicted_bf,
    pose_visibility=1.0,
    mask_confidence=1.0,
    missing_landmarks=0
):

    # =========================================================
    # LOAD ONTOLOGY
    # =========================================================

    onto = get_ontology(
        "ontology/BodyFatOntology.owl"
    ).load()

    with onto:

        # =====================================================
        # CREATE UNIQUE PERSON
        # =====================================================

        uid = str(uuid.uuid4())[:8]

        person = onto.People(f"person_{uid}")

        # =====================================================
        # RAW DATA
        # =====================================================

        person.heightCm = [float(height)]
        person.weightKg = [float(weight)]

        person.chestCm = [float(chest)]
        person.abdomenCm = [float(abdomen)]
        person.hipCm = [float(hip)]

        # =====================================================
        # QUALITY DATA
        # =====================================================

        pose_visibility = float(pose_visibility)
        mask_confidence = float(mask_confidence)
        missing_landmarks = int(missing_landmarks)

        person.poseVisibility = [pose_visibility]
        person.maskConfidence = [mask_confidence]
        person.missingLandmarkCount = [missing_landmarks]

        # =====================================================
        # FEATURE ENGINEERING
        # =====================================================

        bmi = (
            weight / ((height / 100) ** 2)
            if height != 0 else 0
        )

        whr = (
            abdomen / hip
            if hip != 0 else 0
        )

        wthr = (
            abdomen / height
            if height != 0 else 0
        )

        # FIXED FORMULA
        wpa = (
            (abdomen ** 2) / weight
            if weight != 0 else 0
        )

        person.bmiValue = [float(bmi)]
        person.whrValue = [float(whr)]
        person.wthrValue = [float(wthr)]
        person.wpaValue = [float(wpa)]

        # =====================================================
        # PREDICTION
        # =====================================================

        person.predictedBodyFat = [
            float(predicted_bf)
        ]

        # =====================================================
        # CONFIDENCE REASONING
        # =====================================================

        confidence_score = (
            pose_visibility * 0.6 +
            mask_confidence * 0.4
        )

        person.confidenceScore = [
            float(confidence_score)
        ]

        person.qualityScore = [
            float(confidence_score)
        ]

        # =====================================================
        # BMI REASONING
        # =====================================================

        triggered_rules = []

        if bmi < 18.5:

            bmi_class = "Lean"

            triggered_rules.append(
                "RULE_BMI_LEAN"
            )

        elif bmi < 25:

            bmi_class = "Normal"

            triggered_rules.append(
                "RULE_BMI_NORMAL"
            )

        elif bmi < 30:

            bmi_class = "Overweight"

            triggered_rules.append(
                "RULE_BMI_OVERWEIGHT"
            )

        else:

            bmi_class = "Obese"

            triggered_rules.append(
                "RULE_BMI_OBESE"
            )

        # =====================================================
        # FAT LEVEL REASONING
        # =====================================================

        if predicted_bf < 12:

            fat_level = "LowFat"

            triggered_rules.append(
                "RULE_BODYFAT_LOW"
            )

        elif predicted_bf < 20:

            fat_level = "NormalFat"

            triggered_rules.append(
                "RULE_BODYFAT_NORMAL"
            )

        else:

            fat_level = "HighFat"

            triggered_rules.append(
                "RULE_BODYFAT_HIGH"
            )

        # =====================================================
        # IMAGE QUALITY REASONING
        # =====================================================

        quality = "GoodImage"

        quality_reasons = []

        invalid_input = False

        # INVALID
        if (
            abdomen == 0 or
            chest == 0 or
            hip == 0
        ):

            quality = "InvalidInput"

            invalid_input = True

            triggered_rules.append(
                "RULE_INVALID_MEASUREMENTS"
            )

            quality_reasons.append(
                "Missing body measurements detected"
            )

        elif missing_landmarks > 10:

            quality = "InvalidInput"

            invalid_input = True

            triggered_rules.append(
                "RULE_TOO_MANY_MISSING_LANDMARKS"
            )

            quality_reasons.append(
                f"{missing_landmarks} landmarks are missing"
            )

        # LOW QUALITY
        elif (
            pose_visibility < 0.65 or
            mask_confidence < 0.45
        ):

            quality = "LowQualityImage"

            triggered_rules.append(
                "RULE_LOW_IMAGE_QUALITY"
            )

            if pose_visibility < 0.65:

                quality_reasons.append(
                    f"Low pose landmark visibility ({pose_visibility:.2f})"
                )

            if mask_confidence < 0.45:

                quality_reasons.append(
                    f"Unstable body segmentation confidence ({mask_confidence:.2f})"
                )

        # MEDIUM QUALITY
        elif (
            pose_visibility < 0.80 or
            mask_confidence < 0.60 or
            missing_landmarks > 5
        ):

            quality = "MediumImage"

            triggered_rules.append(
                "RULE_MEDIUM_IMAGE_QUALITY"
            )

            quality_reasons.append(
                "Image quality is acceptable but not optimal"
            )

        # GOOD QUALITY
        else:

            quality = "GoodImage"

            triggered_rules.append(
                "RULE_GOOD_IMAGE_QUALITY"
            )

            quality_reasons.append(
                "Pose landmarks and body segmentation are stable"
            )

        # =====================================================
        # FAT DISTRIBUTION REASONING
        # =====================================================

        fat_distribution = []

        # MILD
        if 0.90 <= whr < 0.95:

            fat_distribution.append(
                "MildCentralFat"
            )

            triggered_rules.append(
                "RULE_MILD_CENTRAL_FAT"
            )

        # MEDIUM
        elif 0.95 <= whr < 1.0:

            fat_distribution.append(
                "CentralFatRisk"
            )

            triggered_rules.append(
                "RULE_CENTRAL_FAT_RISK"
            )

        # SEVERE
        elif whr >= 1.0:

            fat_distribution.append(
                "SevereCentralFat"
            )

            triggered_rules.append(
                "RULE_SEVERE_CENTRAL_FAT"
            )

        # WtHR
        if wthr > 0.5:

            fat_distribution.append(
                "AbdominalObesityRisk"
            )

            triggered_rules.append(
                "RULE_ABDOMINAL_OBESITY"
            )

        # =====================================================
        # BODY SHAPE REASONING
        # =====================================================

        if whr < 0.85:

            body_shape = "PearShape"

        elif whr < 0.95:

            body_shape = "BalancedShape"

        else:

            body_shape = "AppleShape"

        # =====================================================
        # SEMANTIC ANOMALY DETECTION
        # =====================================================

        semantic_flags = []

        # Skinny fat
        if bmi < 20 and predicted_bf > 25:

            semantic_flags.append(
                "SkinnyFatAnomaly"
            )

            triggered_rules.append(
                "RULE_SKINNY_FAT"
            )

        # Hidden obesity
        if bmi_class == "Normal" and whr > 0.9:

            semantic_flags.append(
                "HiddenCentralObesity"
            )

            triggered_rules.append(
                "RULE_HIDDEN_OBESITY"
            )

        # Athletic
        if bmi > 25 and predicted_bf < 12:

            semantic_flags.append(
                "AthleticBodyType"
            )

            triggered_rules.append(
                "RULE_ATHLETIC_BODY"
            )

        # Geometry inconsistency
        if chest < abdomen and predicted_bf < 10:

            semantic_flags.append(
                "GeometryInconsistency"
            )

            triggered_rules.append(
                "RULE_GEOMETRY_INCONSISTENCY"
            )

        # Landmark instability
        if (
            pose_visibility < 0.4 and
            missing_landmarks > 10
        ):

            semantic_flags.append(
                "PoseDetectionFailure"
            )

            triggered_rules.append(
                "RULE_POSE_FAILURE"
            )

        # =====================================================
        # SEMANTIC CONSISTENCY SCORE
        # =====================================================

        semantic_score = 100

        semantic_score -= (
            len(semantic_flags) * 15
        )

        semantic_score -= (
            missing_landmarks * 2
        )

        semantic_score = max(
            semantic_score,
            0
        )

        # =====================================================
        # EXPLANATION ENGINE
        # =====================================================

        explanations = []

        explanations.append(
            f"BMI = {bmi:.2f} → {bmi_class}"
        )

        explanations.append(
            f"Predicted Body Fat = {predicted_bf:.2f}% → {fat_level}"
        )

        explanations.append(
            f"Body Shape classified as {body_shape}"
        )

        # WHR
        if whr > 0.90:

            explanations.append(
                f"WHR = {whr:.2f} indicates central fat accumulation"
            )

        # WtHR
        if wthr > 0.50:

            explanations.append(
                f"WtHR = {wthr:.2f} suggests abdominal obesity tendency"
            )

        # Quality reasons
        for q in quality_reasons:

            explanations.append(q)

        # Semantic flags
        for flag in semantic_flags:

            explanations.append(
                f"Semantic anomaly detected: {flag}"
            )

        # =====================================================
        # RECOMMENDATION ENGINE
        # =====================================================

        recommendations = []

        # Fat recommendations
        if (
            "MildCentralFat" in fat_distribution or
            "CentralFatRisk" in fat_distribution
        ):

            recommendations.append(
                "Increase aerobic exercise and monitor waist ratio"
            )

        if "SevereCentralFat" in fat_distribution:

            recommendations.append(
                "Consider medical consultation for visceral fat assessment"
            )

        if "AbdominalObesityRisk" in fat_distribution:

            recommendations.append(
                "Reduce abdominal fat through calorie control"
            )

        # Skinny fat
        if "SkinnyFatAnomaly" in semantic_flags:

            recommendations.append(
                "Increase resistance training and protein intake"
            )

        # Low quality
        if quality != "GoodImage":

            recommendations.append(
                "Retake images under better lighting and standing posture"
            )

        # High fat
        if fat_level == "HighFat":

            recommendations.append(
                "Maintain calorie deficit and increase weekly activity"
            )

        # Low fat
        if fat_level == "LowFat":

            recommendations.append(
                "Maintain balanced nutrition and resistance training"
            )

        # Fallback
        if not recommendations:

            recommendations.append(
                "Maintain current healthy lifestyle"
            )

        # =====================================================
        # FINAL OUTPUT
        # =====================================================

        output = {

            # FEATURES
            "bmi": round(bmi, 2),
            "whr": round(whr, 2),
            "wthr": round(wthr, 2),
            "wpa": round(wpa, 2),

            # REASONING
            "bmi_class": bmi_class,
            "fat_level": fat_level,
            "quality": quality,
            "body_shape": body_shape,

            # PREDICTION
            "predicted_bf": round(
                predicted_bf,
                2
            ),

            "confidence_score": round(
                confidence_score,
                2
            ),
            
            # image quality detail
            "pose_visibility": round(float(pose_visibility), 3),
            "mask_confidence": round(float(mask_confidence), 3),
            "missing_landmarks": int(missing_landmarks),

            # image quality reasons
            "quality_reasons": quality_reasons,

            # SEMANTIC
            "fat_distribution": fat_distribution,
            "semantic_flags": semantic_flags,
            "semantic_score": semantic_score,

            # EXPLAINABILITY
            "triggered_rules": triggered_rules,
            "explanations": explanations,

            # RECOMMENDATION
            "recommendations": recommendations
        }

        return output