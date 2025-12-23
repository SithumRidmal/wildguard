from flask import Flask, request, jsonify, render_template
import onnxruntime as ort
from PIL import Image
import numpy as np
import json, io, os, requests

app = Flask(__name__)

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

MODEL_PATH = "mobilenetv2.onnx"

# -----------------------------
# Load ONNX model ONCE
# -----------------------------
print("Loading MobileNetV2 ONNX model...")
ort_session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)
print("Model loaded successfully.")

# -----------------------------
# Load ImageNet labels
# -----------------------------
with open("data/imagenet_class_index.json") as f:
    class_idx = json.load(f)

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))

    img = np.array(image).astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)   # HWC → CHW
    img = np.expand_dims(img, 0)   # Batch dimension

    return img

# -----------------------------
# Prediction
# -----------------------------
def predict_image(image_bytes):
    img = preprocess_image(image_bytes)
    outputs = ort_session.run(None, {"input": img})[0][0]

    exp = np.exp(outputs - np.max(outputs))
    probs = exp / np.sum(exp)

    top3 = probs.argsort()[-3:][::-1]

    return [{
        "label": class_idx[str(i)][1],
        "confidence": round(float(probs[i] * 100), 2)
    } for i in top3]

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/realtime")
def realtime():
    return render_template("realtime.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    image = request.files["image"].read()
    return jsonify({"predictions": predict_image(image)})

@app.route("/wild-news")
def wild_news():
    url = (
        "https://gnews.io/api/v4/search?"
        "q=wildlife OR endangered OR conservation"
        "&lang=en&sortby=publishedAt&max=6"
        f"&apiKey={NEWS_API_KEY}"
    )
    return jsonify(requests.get(url).json())

with open("templates/animal_data.json") as f:
    animal_data = json.load(f)

@app.route("/animal-details")
def animal_details():
    animal = request.args.get("animal", "").lower()
    return jsonify(animal_data.get(animal, {
        "description": "Details not available.",
        "safety_tips": "Be cautious around wild animals."
    }))

# -----------------------------
# Render entry
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
