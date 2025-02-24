import os

import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.utils import check_and_download_models
from src.data_preprocessing import preprocess_image
from src.Plant_Classification_resnet50.plant_classification_TL import PlantClassifierCNN
from src.Disease_Classification_resnet50.src.disease_model import DiseaseClassifier


# ─────────────────────────────────────────────────────────────────────────────
# 1) Leaf detection function
# ─────────────────────────────────────────────────────────────────────────────
def detect_leaf(image_path: str):
    """
    Detects a leaf in the image using YOLOv8n and extracts it if confidence is above 50%.
    Returns the path to the cropped leaf image ('temp_leaf.jpg'), or None if detection fails.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load YOLOv8 model with weights_only=False to allow model to unpickle.
    YOLO_MODEL_PATH = "src/models/yolov8n_leaf.pt"
    leaf_detector = YOLO(YOLO_MODEL_PATH).to(device)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return None

    # Convert image to RGB (YOLO expects RGB images)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run YOLO inference
    results = leaf_detector(image_rgb)

    if not results or len(results[0].boxes) == 0:
        print("No leaf detected.")
        return None

    # Extract the highest confidence leaf detection
    for box in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, confidence, class_id = box  # bounding box + confidence
        if confidence < 0.70:
            print(f"Detected leaf, but confidence ({confidence:.2f}) is too low.")
            return None

        # Convert coordinates to integers and crop
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        leaf_crop = image_rgb[y1:y2, x1:x2]

        # Convert to PIL image, save the cropped leaf
        cropped_leaf = Image.fromarray(leaf_crop)
        cropped_leaf_path = "temp_leaf.jpg"
        cropped_leaf.save(cropped_leaf_path)

        print(f"Leaf detected with confidence {confidence:.2f} and saved to {cropped_leaf_path}")
        return cropped_leaf_path

    print("No confident leaf detection found.")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2) Helper to load a PyTorch model
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_class, model_path, num_classes, device):
    print(f"Loading model: {model_class.__name__}")
    print(f"Model path: {model_path}")
    print(f"Using device: {device}")

    model = model_class(num_classes)
    try:
        # In PyTorch 2.6+, to forcibly allow pickling custom classes, pass weights_only=False if needed
        # But if your classification .pth is purely weights, no need for that. We'll keep it as is:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        print("Checkpoint loaded successfully!")

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        print("Model loaded and ready!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 3) Plant classification (expects a cropped leaf image!)
# ─────────────────────────────────────────────────────────────────────────────
PLANT_CLASSES = ["Apple", "Corn_(maize)", "Grape", "Pepper_bell", "Potato", "Tomato"]

def classify_plant(cropped_leaf_path: str, model_path: str = "src/models/Plant_Classification.pth") -> str:
    """
    Classify a cropped leaf image using the PlantClassifierCNN model.
    """
    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Load the plant classifier
    model = load_model(
        model_class=PlantClassifierCNN,
        model_path=model_path,
        num_classes=len(PLANT_CLASSES),
        device=device
    )

    # Preprocess the cropped leaf image
    image_tensor = preprocess_image(cropped_leaf_path).to(device)

    # Inference
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        predicted_prob = probabilities[0, predicted_idx].item()

    plant_name = PLANT_CLASSES[predicted_idx]
    if predicted_prob < 0.50:
        print(f"Low confidence prediction: {predicted_prob:.2f}")
        plant_name = "Unknown"

    print(f"Predicted Plant: {plant_name} (Confidence: {predicted_prob:.2f})")
    return plant_name


# ─────────────────────────────────────────────────────────────────────────────
# 4) Disease classification (expects a cropped leaf image!)
# ─────────────────────────────────────────────────────────────────────────────
def classify_disease(cropped_leaf_path: str, plant_type: str) -> str:
    """
    Classify the disease on a cropped leaf image using the DiseaseClassifier model.
    """
    if plant_type == "Unknown":
        return "Unknown"

    disease_labels = {
        "Apple": ["Apple Scab", "Apple Black Rot", "Cedar Apple Rust", "Apple Healthy"],
        "Corn_(maize)": ["Cercospora_leaf_spot Gray_leaf_spot", "Common Rush", "Northern_Leaf_Blight", "Healthy"],
        "Grape": ["Black Rot", "Esca_(Black_Measles)", "Leaf_blight_(Isariopsis_Leaf_Spot)", "Healthy"],
        "Pepper_bell": ["Bacterial_Spot", "Healthy"],
        "Potato": ["Early_Blight", "Late_Blight", "Healthy"],
        "Tomato": [
            "Bacterial_Spot", "Early_Blight", "Late_Blight", "Leaf_Mold",
            "Septoria_leaf_spot", "Spider_Mites", "Target_Spot",
            "Yellow_Leaf_Curl_Virus", "Mosaic_Virus", "Healthy"
        ],
    }
    class_labels = disease_labels.get(plant_type, ["Unknown"])  # default to ["Unknown"] if not found

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Dynamically determine model file based on plant name
    model_path = f"src/models/{plant_type}_Disease_Classification.pth"
    if not os.path.exists(model_path):
        print(f"No disease model found for {plant_type}, returning 'Unknown'")
        return "Unknown"

    # Load disease classifier
    model = load_model(
        model_class=DiseaseClassifier,
        model_path=model_path,
        num_classes=len(class_labels),
        device=device
    )

    # Preprocess the cropped leaf image
    image_tensor = preprocess_image(cropped_leaf_path).to(device)

    # Inference
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        predicted_prob = probabilities[0, predicted_idx].item()

    disease_prediction = class_labels[predicted_idx]
    if predicted_prob < 0.50:
        print(f"Low confidence disease prediction: {predicted_prob:.2f}")
        disease_prediction = "Unknown"

    print(f"Predicted Disease: {disease_prediction} (Confidence: {predicted_prob:.2f})")
    return disease_prediction