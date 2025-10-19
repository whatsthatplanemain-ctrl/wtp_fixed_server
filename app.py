from flask import Flask, request, jsonify
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

app = Flask(__name__)

# Path to your TorchScript model
MODEL_PATH = "model_aircraft.pt"
model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()

# Custom transform: pad to square, then resize
def pad_to_square(img, fill=(0, 0, 0)):
    w, h = img.size
    max_side = max(w, h)
    pad_left = (max_side - w) // 2
    pad_top = (max_side - h) // 2
    pad_right = max_side - w - pad_left
    pad_bottom = max_side - h - pad_top
    return F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=fill)

transform = transforms.Compose([
    transforms.Lambda(lambda img: pad_to_square(img)),  # pad to square
    transforms.Resize((224, 224)),                      # downscale to 224x224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Your dataset classes
labels = [
    "737",
    "747",
    "757",
    "777",
    "787",
    "a320",
    "a330",
    "a330-beluga",
    "a340",
    "a350",
    "a380",
    "an_124",
    "cessna172",
    "eurofighter_typhoon",
    "not_planes"
]

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Aircraft recognition API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        image = Image.open(file.stream).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            class_id = predicted.item()
            class_name = labels[class_id] if class_id < len(labels) else str(class_id)

        return jsonify({"class_id": class_id, "class_name": class_name})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Local run
    app.run(host="0.0.0.0", port=8080, debug=True)
