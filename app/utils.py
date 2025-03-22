import cv2
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize for model input
    img = img / 255.0  # Normalize
    return np.expand_dims(img, axis=0)  # Add batch dimension

def generate_report(disease, result, doctor_note):
    report = f"""
    Medical Report
    --------------
    Disease: {disease.capitalize()}
    Prediction Result: {result}
    Doctor's Note: {doctor_note}
    """
    return report
