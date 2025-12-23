from flask import Flask, request, jsonify, render_template
import requests
import onnxruntime as ort
from PIL import Image
import numpy as np
import json, io

app = Flask(__name__)

NEWS_API_KEY = "454588b35e7f0d25e8cd5b2f32d96677"

# Load ONNX model
ort_session = ort.InferenceSession("./vgg16.onnx")

with open("./data/imagenet_class_index.json") as f:
    class_idx = json.load(f)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    img_data = np.array(image).astype(np.float32)
    # Normalize according to ImageNet
    img_data = img_data / 255.0
    img_data = img_data.transpose(2, 0, 1)  # HWC -> CHW
    img_data = np.expand_dims(img_data, axis=0)  # Add batch dimension
    return img_data

def predict_image(image_bytes):
    img = preprocess_image(image_bytes)
    outputs = ort_session.run(None, {"input": img})[0][0]  # Get first batch
    probs = np.exp(outputs) / np.sum(np.exp(outputs))  # Softmax
    top3_idx = probs.argsort()[-3:][::-1]

    return [{
        "label": class_idx[str(idx)][1],
        "confidence": round(float(probs[idx] * 100), 2)
    } for idx in top3_idx]

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
        "q=wildlife OR \"wild animals\" OR endangered OR conservation"
        "&lang=en"
        "&sortby=publishedAt"
        "&max=6"
        f"&apiKey={NEWS_API_KEY}"
    )

    response = requests.get(url)
    return jsonify(response.json())

with open("./templates/animal_data.json") as f:
    animal_data = json.load(f)

@app.route("/animal-details")
def animal_details():
    animal = request.args.get("animal", "").lower()
    if animal in animal_data:
        return jsonify(animal_data[animal])
    else:
        return jsonify({
            "description": "Details not available.",
            "safety_tips": "Be cautious and keep distance from unknown animals."
        })

if __name__ == "__main__":
    app.run()
