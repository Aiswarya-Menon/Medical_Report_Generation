from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import cv2
import os
from dotenv import load_dotenv
import torch
import torchvision.transforms as transforms
from PIL import Image
import tensorflow as tf
import google.generativeai as genai  # Gemini API
import markdown
from bs4 import BeautifulSoup

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

brain_tumor_model = BrainTumorModel(num_classes=4)
brain_tumor_model.load_state_dict(torch.load("models/brain_tumor_model.pth", map_location=torch.device('cpu')))
brain_tumor_model.eval()

# Load PyTorch Alzheimer's Model
class AlzheimerModel(torch.nn.Module):
    def __init__(self, num_classes=4):
        super(AlzheimerModel, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

alzheimers_model = AlzheimerModel(num_classes=4)
alzheimers_model.load_state_dict(torch.load("models/alzheimers_model.pth", map_location=torch.device('cpu')))
alzheimers_model.eval()

# Class labels for each disease
class_labels = {
    "pneumonia": ['NORMAL', 'PNEUMONIA'],
    "brain_tumor": ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor'],
    "alzheimers": ['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very Mild Dementia']
}

# Define preprocessing for brain tumor and Alzheimer's models
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Configure Gemini API

load_dotenv()

# Get the API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Replace with actual API key
genai.configure(api_key=GEMINI_API_KEY)
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # Ensure GOOGLE_API_KEY is set in your environment
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

    image_path = os.path.join("static/uploads", file.filename)
    file.save(image_path)

    if disease == "pneumonia":
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (120, 120))
        img_array = np.expand_dims(img, axis=0) / 255.0

        prediction = pneumonia_model.predict(img_array)
        predicted_class = class_labels[disease][int(prediction[0][0] > 0.5)]

    elif disease == "brain_tumor" or disease == "alzheimers":
        img = Image.open(image_path).convert('RGB')
        img_tensor = image_transforms(img).unsqueeze(0)

        model = brain_tumor_model if disease == "brain_tumor" else alzheimers_model

        with torch.no_grad():
            outputs = model(img_tensor)
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
    image_path = data.get("image", "")

    if not disease or not prediction:
        return jsonify({"error": "Disease and prediction result required"})

    disease_prompts = {
        "pneumonia": "Provide a detailed medical report for a chest X-ray indicating {}.",
        "brain_tumor": "Analyze the MRI scan and generate a comprehensive report for a detected {}.",
        "alzheimers": "Analyze the brain MRI and generate a comprehensive report for a patient with {}."
    }

    prompt_template = disease_prompts.get(disease, "Generate a professional medical report for {}. Dont include incomplete data. Make the report a concise one")
    prompt = prompt_template.format(prediction) + " Ensure the report is concise, professional, and contains only complete information."

    if doctor_notes:
        prompt += f"\nDoctor's Notes: {doctor_notes}"

    response = gemini_model.generate_content(prompt)

    raw_report = response.text.strip() if response and response.text else "Report generation failed."

    html_report = markdown.markdown(raw_report)
    soup = BeautifulSoup(html_report, "html.parser")
    plain_text_report = soup.get_text()

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