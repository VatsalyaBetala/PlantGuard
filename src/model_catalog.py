import os
import re
from typing import Dict, List

PLANT_CLASSES: List[str] = [
    "Apple",
    "Corn_(maize)",
    "Grape",
    "Pepper_bell",
    "Potato",
    "Tomato",
]

DISEASE_LABELS: Dict[str, List[str]] = {
    "Apple": ["Apple Scab", "Apple Black Rot", "Cedar Apple Rust", "Apple Healthy"],
    "Corn_(maize)": [
        "Cercospora_leaf_spot Gray_leaf_spot",
        "Common Rush",
        "Northern_Leaf_Blight",
        "Healthy",
    ],
    "Grape": ["Black Rot", "Esca_(Black_Measles)", "Leaf_blight_(Isariopsis_Leaf_Spot)", "Healthy"],
    "Pepper_bell": ["Bacterial_Spot", "Healthy"],
    "Potato": ["Early_Blight", "Late_Blight", "Healthy"],
    "Tomato": [
        "Bacterial_Spot",
        "Early_Blight",
        "Late_Blight",
        "Leaf_Mold",
        "Septoria_leaf_spot",
        "Spider_Mites",
        "Target_Spot",
        "Yellow_Leaf_Curl_Virus",
        "Mosaic_Virus",
        "Healthy",
    ],
}

LEGACY_PLANT_MODEL_FILENAME = "Plant_Classification.pth"
LEGACY_DISEASE_MODEL_FILENAMES: Dict[str, str] = {
    plant: f"{plant}_Disease_Classification.pth" for plant in DISEASE_LABELS
}
LEGACY_RESNET_WEIGHTS = "resnet50_weights.pth"
LEGACY_YOLO_WEIGHTS = "yolov8n_leaf.pt"


def get_backend_name() -> str:
    return os.getenv("MODEL_NAME", "cnn_resnet50")


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def plant_model_name(backend: str | None = None) -> str:
    backend = backend or get_backend_name()
    return f"{backend}_plant"


def disease_model_name(plant_type: str, backend: str | None = None) -> str:
    backend = backend or get_backend_name()
    return f"{backend}_disease_{slugify(plant_type)}"


def available_model_names(backend: str | None = None) -> List[str]:
    backend = backend or get_backend_name()
    models = [plant_model_name(backend)]
    models.extend(disease_model_name(plant, backend) for plant in DISEASE_LABELS)
    return models
