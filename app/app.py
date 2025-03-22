from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import cv2
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import tensorflow as tf
import google.generativeai as genai  # Gemini API
import markdown
from bs4 import BeautifulSoup

from flask import send_file
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

app = Flask(__name__)

# Load TensorFlow Pneumonia Model
pneumonia_model = tf.keras.models.load_model("models/pneumonia_model.h5")

# Load PyTorch Brain Tumor Model
class BrainTumorModel(torch.nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorModel, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

# Initialize model and load weights
brain_tumor_model = BrainTumorModel(num_classes=4)
brain_tumor_model.load_state_dict(torch.load("models/brain_tumor_model.pth", map_location=torch.device('cpu')))
brain_tumor_model.eval()

# Class labels for each disease
class_labels = {
    "pneumonia": ['NORMAL', 'PNEUMONIA'],
    "brain_tumor": ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
}

# Define preprocessing for brain tumor model
brain_tumor_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyBOWIQnRV6R3a6tMgyrsofph-4nQSNSf8k"  # Replace with actual API key
genai.configure(api_key=GEMINI_API_KEY)

# Select the correct Gemini model
gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")  # Use an available model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or 'disease' not in request.form:
        return jsonify({"error": "File and disease type are required"})

    file = request.files['file']
    disease = request.form['disease']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    # Save the uploaded file
    image_path = os.path.join("static/uploads", file.filename)
    file.save(image_path)

    # Load and preprocess the image
    if disease == "pneumonia":
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (120, 120))
        img_array = np.expand_dims(img, axis=0) / 255.0  # Normalize

        # Predict using Pneumonia model
        prediction = pneumonia_model.predict(img_array)
        predicted_class = class_labels[disease][int(prediction[0][0] > 0.5)]

    elif disease == "brain_tumor":
        img = Image.open(image_path).convert('RGB')
        img = brain_tumor_transforms(img)
        img = img.unsqueeze(0)  # Add batch dimension

        # Predict using Brain Tumor model
        with torch.no_grad():
            outputs = brain_tumor_model(img)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_labels[disease][predicted.item()]

    else:
        return jsonify({"error": "Invalid disease selection"})

    return jsonify({"disease": disease, "prediction": predicted_class, "image": image_path})

@app.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.json
    disease = data.get("disease", "")
    prediction = data.get("prediction", "")
    doctor_notes = data.get("doctor_notes", "")

    if not disease or not prediction:
        return jsonify({"error": "Disease and prediction result required"})

    # Upload image to Gemini
    image_path = data.get("image", "")
    gemini_image = genai.upload_file(image_path) if image_path else None

    # Disease-specific prompts
    disease_prompts = {
        "pneumonia": "Provide a detailed medical report for a chest X-ray indicating {}.",
        "brain_tumor": "Analyze the MRI scan and generate a comprehensive report for a detected {}."
    }
    
    # Construct prompt dynamically
    prompt_template = disease_prompts.get(disease, "Generate a professional medical report for {}.")
    prompt = prompt_template.format(prediction)

    # Include doctor's notes if available
    if doctor_notes:
        prompt += f"\nDoctor's Notes: {doctor_notes}"

    # Generate report using Gemini
    response = gemini_model.generate_content([prompt, gemini_image] if gemini_image else prompt)

    # Ensure valid report generation
    raw_report = response.text.strip() if response and response.text else "Report generation failed."

    # Convert Markdown to HTML, then extract plain text
    html_report = markdown.markdown(raw_report)
    soup = BeautifulSoup(html_report, "html.parser")
    plain_text_report = soup.get_text()

    # Save to a file
    report_path = "static/generated_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(plain_text_report)

    return jsonify({"report": plain_text_report, "report_path": report_path})

@app.route('/view_report')
def view_report():
    report_path = "static/generated_report.txt"

    if not os.path.exists(report_path):
        return render_template("view_report.html", report="No report found. Please generate a report first.")

    with open(report_path, "r", encoding="utf-8") as f:
        report_content = f.read()

    return render_template("view_report.html", report=report_content)


@app.route('/download_report')
def download_report():
    report_path = "static/generated_report.txt"
    if os.path.exists(report_path):
        return send_file(report_path, as_attachment=True)
    else:
        return jsonify({"error": "Report not found"})

if __name__ == '__main__':
    app.run(debug=True)
