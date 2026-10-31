import os
import io
import json
import base64
import numpy as np
import cv2
import joblib
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

from ml.features import preprocess_image, extract_combined_features, extract_biomechanical_metrics, generate_tremor_heatmap
from ml.train_cnn import LightweightCNN

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Load Random Forest models
print("Loading Random Forest models...")
rf_models = {
    "spiral": joblib.load(os.path.join(MODELS_DIR, "rf_spiral.joblib")),
    "wave": joblib.load(os.path.join(MODELS_DIR, "rf_wave.joblib"))
}

# Load PyTorch CNN models
print("Loading CNN models...")
device = torch.device("cpu")
cnn_models = {}
for task in ["spiral", "wave"]:
    model = LightweightCNN()
    model_path = os.path.join(MODELS_DIR, f"cnn_{task}.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    cnn_models[task] = model

# Load Metrics
rf_metrics = {}
cnn_metrics = {}
if os.path.exists(os.path.join(MODELS_DIR, "rf_metrics.json")):
    with open(os.path.join(MODELS_DIR, "rf_metrics.json")) as f:
        rf_metrics = json.load(f)
if os.path.exists(os.path.join(MODELS_DIR, "cnn_metrics.json")):
    with open(os.path.join(MODELS_DIR, "cnn_metrics.json")) as f:
        cnn_metrics = json.load(f)

cnn_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def decode_image_bytes(raw_bytes):
    """
    Safely decodes image bytes using OpenCV with fallback to PIL.
    Supports PNG, JPG, JPEG, TIFF, BMP, WEBP, and RGBA/Grayscale formats.
    """
    if not raw_bytes or len(raw_bytes) == 0:
        return None
    try:
        data = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is not None and img.size > 0:
            return img
    except Exception as e:
        print(f"OpenCV decode error: {e}")

    try:
        pil_im = Image.open(io.BytesIO(raw_bytes))
        return np.array(pil_im)
    except Exception as e:
        print(f"PIL decode error: {e}")

    return None

def evaluate_drawing(img_array, task="spiral", model_type="cnn"):
    """
    Perform full diagnostic evaluation:
    1. Preprocess image
    2. Extract biomechanical biomarkers
    3. Generate tremor heatmap
    4. Run ML model inference
    5. Calculate clinical UPDRS risk stage
    """
    preprocessed = preprocess_image(img_array)
    biomech = extract_biomechanical_metrics(preprocessed)
    heatmap_overlay = generate_tremor_heatmap(preprocessed)

    # Encode heatmap to base64
    _, buffer = cv2.imencode(".jpg", heatmap_overlay)
    heatmap_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

    # Encode preprocessed image to base64 for inspection
    _, buffer_orig = cv2.imencode(".jpg", preprocessed)
    preprocessed_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer_orig).decode("utf-8")

    pred_class = 0
    prob_pd = 0.0
    prob_healthy = 1.0

    if model_type == "cnn":
        pil_img = Image.fromarray(preprocessed)
        tensor_img = cnn_transform(pil_img).unsqueeze(0)
        with torch.no_grad():
            outputs = cnn_models[task](tensor_img)
            probs = torch.softmax(outputs, dim=1).squeeze().numpy()
            prob_healthy = float(probs[0])
            prob_pd = float(probs[1])
            pred_class = int(np.argmax(probs))
    else:  # "rf"
        feat_vector, _ = extract_combined_features(preprocessed)
        feat_vector = feat_vector.reshape(1, -1)
        probs = rf_models[task].predict_proba(feat_vector)[0]
        prob_healthy = float(probs[0])
        prob_pd = float(probs[1])
        pred_class = int(np.argmax(probs))

    # UPDRS / Hoehn & Yahr stage estimation based on probability and tremor index
    if pred_class == 0:
        stage = "Stage 0 (No Motor Impairment)"
        severity = "Normal / Controlled"
        recommendation = "No typical parkinsonian kinetic tremor patterns detected. Periodic annual motor wellness screening advised."
    else:
        tremor_score = biomech["tremor_index"]
        if tremor_score < 35.0:
            stage = "Early Stage 1 (Unilateral / Mild Tremor Signs)"
            severity = "Mild Kinetic Tremor"
            recommendation = "Subtle line hesitations and micro-oscillations detected. Recommend comprehensive clinical neurological examination and DaTscan if symptoms persist."
        elif tremor_score < 65.0:
            stage = "Stage 1-2 (Mild-to-Moderate Bilateral Motor Rigidity)"
            severity = "Moderate Tremor Irregularity"
            recommendation = "Noticeable frequency anomalies and stroke curvature deviations detected. Consult a movement disorder specialist for UPDRS motor evaluation."
        else:
            stage = "Stage 2+ (Pronounced Micro-Tremor & Motor Rigidity)"
            severity = "Severe Tremor / Line Instability"
            recommendation = "Significant stroke discontinuities, high spectral variance, and angular deviations observed. Immediate specialist neurological assessment recommended."

    return {
        "status": "success",
        "task": task,
        "model_used": "Convolutional Neural Network (CNN)" if model_type == "cnn" else "Random Forest (Biomechanical + HOG)",
        "model_type": model_type,
        "prediction": "Parkinson's Disease Detected" if pred_class == 1 else "Healthy Control",
        "is_parkinson": bool(pred_class == 1),
        "confidence_pct": round(float(max(prob_pd, prob_healthy) * 100), 1),
        "probability_parkinson": round(prob_pd * 100, 1),
        "probability_healthy": round(prob_healthy * 100, 1),
        "clinical_assessment": {
            "stage": stage,
            "severity": severity,
            "recommendation": recommendation
        },
        "biomarkers": {
            "tremor_index": biomech["tremor_index"],
            "curvature_variance": biomech["curvature_variance"],
            "jaggedness_ratio": biomech["jaggedness_ratio"],
            "spectral_energy_ratio": biomech["spectral_energy_ratio"],
            "stroke_width_std": biomech["stroke_width_std"],
            "gradient_entropy": biomech["grad_entropy"],
            "line_smoothness_score": round(max(0, 100.0 - biomech["tremor_index"]), 1)
        },
        "visuals": {
            "preprocessed_image": preprocessed_b64,
            "tremor_heatmap": heatmap_b64
        }
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        task = request.form.get("task", "spiral")
        model_type = request.form.get("model_type", "cnn")

        img_array = None

        # 1. Check if multipart file uploaded
        if "file" in request.files:
            file = request.files["file"]
            if file and file.filename != "":
                raw_bytes = file.read()
                img_array = decode_image_bytes(raw_bytes)

        # 2. Check base64 form parameter
        if img_array is None and "image_data" in request.form:
            b64_str = request.form["image_data"].strip()
            if b64_str:
                if "," in b64_str:
                    b64_str = b64_str.split(",")[1]
                try:
                    raw_bytes = base64.b64decode(b64_str)
                    img_array = decode_image_bytes(raw_bytes)
                except Exception as b64_err:
                    print(f"Base64 error: {b64_err}")

        # 3. Check JSON payload
        if img_array is None and request.is_json:
            json_data = request.get_json(silent=True) or {}
            task = json_data.get("task", task)
            model_type = json_data.get("model_type", model_type)
            b64_str = json_data.get("image_data", "").strip()
            if b64_str:
                if "," in b64_str:
                    b64_str = b64_str.split(",")[1]
                try:
                    raw_bytes = base64.b64decode(b64_str)
                    img_array = decode_image_bytes(raw_bytes)
                except Exception as b64_err:
                    print(f"JSON Base64 error: {b64_err}")

        if img_array is None:
            return jsonify({
                "status": "error",
                "message": "No valid image data could be parsed. Please select or drag a valid PNG, JPG, or JPEG drawing image."
            }), 400

        result = evaluate_drawing(img_array, task=task, model_type=model_type)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Diagnostic processing error: {str(e)}"
        }), 500

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    return jsonify({
        "status": "success",
        "random_forest": rf_metrics,
        "cnn": cnn_metrics
    })

@app.route("/api/samples", methods=["GET"])
def get_samples():
    """
    Returns verified clinical dataset samples for 1-click evaluation.
    """
    samples = []
    for task in ["spiral", "wave"]:
        for category in ["healthy", "parkinson"]:
            test_dir = os.path.join(DATASET_DIR, task, "testing", category)
            if os.path.exists(test_dir):
                files = sorted(os.listdir(test_dir))[:4]
                for idx, fname in enumerate(files):
                    samples.append({
                        "id": f"{task}_{category}_{idx}",
                        "task": task,
                        "category": category,
                        "label": f"{'Healthy Control' if category == 'healthy' else 'Parkinson Patient'} ({task.title()}) #{idx+1}",
                        "url": f"/api/sample-image/{task}/testing/{category}/{fname}"
                    })
    return jsonify({"status": "success", "samples": samples})

@app.route("/api/sample-image/<task>/<split>/<category>/<filename>")
def serve_sample(task, split, category, filename):
    safe_dir = os.path.join(DATASET_DIR, task, split, category)
    return send_from_directory(safe_dir, filename)

@app.route("/sample_images/<path:filename>")
def serve_sample_image(filename):
    samples_dir = os.path.join(BASE_DIR, "sample_images")
    return send_from_directory(samples_dir, filename)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
