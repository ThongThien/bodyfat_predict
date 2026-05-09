import streamlit as st

def show_info_page_v5():

    st.title("Body Fat AI System")

    st.markdown("""
### Overview

Body Fat AI is a hybrid health-analysis system that estimates body fat percentage from only two body images and several basic physical measurements.  
The application combines Computer Vision, Geometry-based body measurement, Machine Learning prediction, and Ontology Reasoning into a single pipeline.

Instead of predicting directly from raw images, the system first extracts meaningful body measurements such as Chest, Abdomen, and Hip circumference.  
These measurements are then transformed into health-related body indices before being analyzed by the AI model.

The overall workflow is:

**Images → Body Measurement Extraction → Feature Engineering → AI Prediction → Ontology Reasoning → Explanation & Recommendation**

---

### Computer Vision Pipeline

The image-processing module is built using MediaPipe Pose and Selfie Segmentation.

The segmentation model separates the human body from the background to create a body mask, while the pose model detects important body landmarks such as shoulders, hips, heels, and torso positions.  
These landmarks are used to estimate body proportions and identify standardized measurement regions.

The system then scans both the front and side body images to calculate body width and body depth.  
Instead of using a single fixed line, multiple scan positions are tested to reduce errors caused by posture variation, imperfect landmark detection, or slight body rotation.

To estimate circumference values, the body cross-section is approximated as an ellipse.  
An ellipse-based geometric formula is then applied to estimate Chest, Abdomen, and Hip measurements in centimeters.

Finally, pixel measurements are converted into real-world scale using the user's actual height.

---

### AI Prediction Model

The prediction engine uses a tuned Random Forest Regressor trained on body composition data.

The model does not use raw images directly.  
Instead, it analyzes engineered body features that are strongly related to fat distribution and obesity patterns.

The final input features are:

- Weight
- Chest circumference
- Abdomen circumference
- Hip circumference
- WtHR (Waist-to-Height Ratio)
- WHR (Waist-to-Hip Ratio)
- W_per_A (Waist Power Index)

The dataset contains 195 labeled samples and was optimized specifically for body fat estimation tasks.

Current performance metrics:

- R² Score: approximately 0.82
- MAE: approximately 2.3%
- RMSE: approximately 2.9%

These results indicate relatively stable prediction quality for a lightweight non-contact body analysis system.

---

### Ontology Reasoning Layer

Beyond AI prediction, the system also includes an Ontology-based reasoning module developed with OWL and Owlready2.

This layer allows the application to perform semantic interpretation instead of displaying only raw numerical outputs.

The ontology engine can:

- classify BMI levels
- identify body fat categories
- detect central fat accumulation
- validate image quality
- identify semantic inconsistencies
- generate explanations
- provide health-oriented recommendations

For example, the system can detect situations such as:

- normal BMI but high abdominal fat risk
- low BMI with unexpectedly high body fat
- low-quality images that may reduce prediction reliability

This reasoning layer improves explainability and makes the system easier for users to understand.

---

### Reliability and Limitations

The system is designed as a practical and accessible body fat estimation tool rather than a replacement for medical equipment such as DEXA or professional InBody scanners.

Under standard image conditions, the average prediction error is usually around 2–3% body fat.  
However, accuracy can decrease when images contain poor lighting, loose clothing, incorrect standing posture, body occlusion, or incomplete segmentation.

For best results, users should:

- capture clear front and side body images
- stand upright in a standardized pose
- avoid oversized clothing
- ensure good lighting and background contrast

---

### Conclusion

Body Fat AI demonstrates how Computer Vision, Geometry, Machine Learning, and Ontology Reasoning can be combined into an explainable health-analysis system.

The project focuses not only on prediction accuracy, but also on interpretability, semantic validation, and user understanding.

The system is suitable for fitness tracking, personal health monitoring, educational research, and AI-based healthcare prototypes.
""")