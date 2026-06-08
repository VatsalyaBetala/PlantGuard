import os

import cv2
import torch
from PIL import Image
from ultralytics import YOLO

from plant_disease.model_artifacts import resolve_shared_file
from plant_disease.model_catalog import (
    DISEASE_LABELS,
    disease_model_name,
    get_backend_name,
    plant_model_name,
)
from plant_disease.model_registry import get_model

CROPS_DIR = "uploads/_crops"


def detect_leaf(image_path: str):
    """
    Detect a leaf with YOLO and save a cropped leaf image.

    Returns the cropped image path, or None if detection fails.
    The crop is written to a unique path derived from the source image, so
    concurrent uploads do not clobber each other and Grad-CAM outputs end up
    with one file per prediction.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_model_path = resolve_shared_file("yolov8n_leaf.pt")
    leaf_detector = YOLO(str(yolo_model_path)).to(device)

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = leaf_detector(image_rgb)

    if not results or len(results[0].boxes) == 0:
        print("No leaf detected.")
        return None

    os.makedirs(CROPS_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    cropped_leaf_path = os.path.join(CROPS_DIR, f"{base}_crop.jpg")

    for box in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, confidence, _class_id = box
        if confidence < 0.70:
            print(f"Detected leaf, but confidence ({confidence:.2f}) is too low.")
            return None

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        leaf_crop = image_rgb[y1:y2, x1:x2]
        Image.fromarray(leaf_crop).save(cropped_leaf_path)

        print(f"Leaf detected with confidence {confidence:.2f} and saved to {cropped_leaf_path}")
        return cropped_leaf_path

    print("No confident leaf detection found.")
    return None


def classify_plant_with_details(cropped_leaf_path: str) -> dict:
    """Classify a cropped leaf and return label, confidence, and top-k data."""
    backend = get_backend_name()
    model = get_model(plant_model_name(backend))
    image_tensor = model.preprocess(cropped_leaf_path)
    output = model.predict(image_tensor)
    result = model.postprocess(output)

    label = result["label"]
    if result["confidence"] < 0.50:
        label = "Unknown"

    return {
        "label": label,
        "confidence": result["confidence"],
        "topk": result.get("topk", []),
    }


def classify_plant(cropped_leaf_path: str) -> str:
    """Compatibility wrapper that returns only the predicted plant label."""
    return classify_plant_with_details(cropped_leaf_path)["label"]


def classify_disease_with_details(cropped_leaf_path: str, plant_type: str) -> dict:
    """Classify disease and return label, confidence, and top-k data."""
    if plant_type == "Unknown" or plant_type not in DISEASE_LABELS:
        return {"label": "Unknown", "confidence": None, "topk": []}

    backend = get_backend_name()
    model_name = disease_model_name(plant_type, backend)

    try:
        model = get_model(model_name)
    except FileNotFoundError:
        print(f"No disease model found for {plant_type}, returning 'Unknown'")
        return {"label": "Unknown", "confidence": None, "topk": []}

    image_tensor = model.preprocess(cropped_leaf_path)
    output = model.predict(image_tensor)
    result = model.postprocess(output)

    label = result["label"]
    if result["confidence"] < 0.50:
        print(f"Low confidence disease prediction: {result['confidence']:.2f}")
        label = "Unknown"

    print(f"Predicted Disease: {label} (Confidence: {result['confidence']:.2f})")
    return {
        "label": label,
        "confidence": result["confidence"],
        "topk": result.get("topk", []),
    }


def classify_disease(cropped_leaf_path: str, plant_type: str) -> str:
    """Compatibility wrapper that returns only the predicted disease label."""
    return classify_disease_with_details(cropped_leaf_path, plant_type)["label"]


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
