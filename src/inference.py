import os

import cv2
import torch
from PIL import Image
from ultralytics import YOLO

from src.model_artifacts import resolve_shared_file
from src.model_catalog import DISEASE_LABELS, disease_model_name, get_backend_name, plant_model_name
from src.model_registry import get_model


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
    yolo_model_path = resolve_shared_file("yolov8n_leaf.pt")
    leaf_detector = YOLO(str(yolo_model_path)).to(device)

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
# 2) Plant classification (expects a cropped leaf image!)
# ─────────────────────────────────────────────────────────────────────────────

def classify_plant(cropped_leaf_path: str) -> str:
    """
    Classify a cropped leaf image using the configured model adapter.
    """
    backend = get_backend_name()
    model_name = plant_model_name(backend)

    model = get_model(model_name)
    image_tensor = model.preprocess(cropped_leaf_path)
    output = model.predict(image_tensor)
    result = model.postprocess(output)

    plant_name = result["label"]
    if result["confidence"] < 0.50:
        plant_name = "Unknown"

    return plant_name


# ─────────────────────────────────────────────────────────────────────────────
# 3) Disease classification (expects a cropped leaf image!)
# ─────────────────────────────────────────────────────────────────────────────

def classify_disease(cropped_leaf_path: str, plant_type: str) -> str:
    """
    Classify the disease on a cropped leaf image using the configured model adapter.
    """
    if plant_type == "Unknown":
        return "Unknown"

    if plant_type not in DISEASE_LABELS:
        return "Unknown"

    backend = get_backend_name()
    model_name = disease_model_name(plant_type, backend)

    try:
        model = get_model(model_name)
    except FileNotFoundError:
        print(f"No disease model found for {plant_type}, returning 'Unknown'")
        return "Unknown"

    image_tensor = model.preprocess(cropped_leaf_path)
    output = model.predict(image_tensor)
    result = model.postprocess(output)

    disease_prediction = result["label"]
    if result["confidence"] < 0.50:
        print(f"Low confidence disease prediction: {result['confidence']:.2f}")
        disease_prediction = "Unknown"

    print(
        f"Predicted Disease: {disease_prediction} (Confidence: {result['confidence']:.2f})"
    )
    return disease_prediction


# ─────────────────────────────────────────────────────────────────────────────
# 4) Disease explanation (Grad-CAM)
# ─────────────────────────────────────────────────────────────────────────────

def explain_disease(cropped_leaf_path: str, plant_type: str):
    if plant_type == "Unknown":
        return None

    backend = get_backend_name()
    model_name = disease_model_name(plant_type, backend)
    try:
        model = get_model(model_name)
    except FileNotFoundError:
        return None

    return model.explain(cropped_leaf_path, {"label": plant_type})
