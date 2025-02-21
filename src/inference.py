import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from src.data_preprocessing import preprocess_image  # Assumes this is your preprocessing function
from src.Plant_Classification_resnet50.plant_classification_TL import PlantClassifierCNN
from src.Disease_Classification_resnet50.src.disease_model import DiseaseClassifier

# Define your plant and disease class labels.
PLANT_CLASSES = ["Apple", "Corn_(maize)", "Grape", "Pepper,_bell", "Potato", "Tomato"]


def load_model(model_class, model_path, num_classes, device):
    """
    Load a model given its class, file path to weights, number of classes, and device.
    """
    model = model_class(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def classify_plant(image_path: str, model_path: str="src/Plant_Classification_resnet50/models/Plant_Classification.pth") -> str:
    """
    Classify a plant image using the PlantClassifierCNN model.
    Loads model weights from the specified file path.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available()
                          else "mps:0" if torch.backends.mps.is_available()
    else "cpu")
    # Load the plant classifier using the file path.
    model = load_model(PlantClassifierCNN, model_path, num_classes=len(PLANT_CLASSES), device=device)

    # Preprocess the image.
    image_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()

    plant_name = PLANT_CLASSES[predicted_idx]
    print(f"Predicted Plant: {plant_name} (Index: {predicted_idx})")
    return plant_name


def classify_disease(image_path: str, plant_type: str) -> str:
    """
    Classify the disease on a plant image using the DiseaseClassifier model.
    Loads model weights from the specified file path.
    """
    disease_labels = {
        "Apple": ["Apple Scab", "Apple Black Rot", "Cedar Apple Rust", "Apple Healthy"],
        "Corn_(maize)": ["Cercospora_leaf_spot Gray_leaf_spot", "Common Rush", "Northern_Leaf_Blight", "Healthy"],
        "Grape": ["Black Rot", "Esca_(Black_Measles)", "Leaf_blight_(Isariopsis_Leaf_Spot)", "Healthy"],
        "Pepper,_bell": ["Bacterial_Spot", "Healthy"],
        "Potato": ["Early_Blight", "Late_Blight", "Healthy"],
        "Tomato": ["Bacterial_Spot", "Early_Blight", "Late_Blight", "Leaf_Mold", "Septoria_leaf_spot",
                   "Spider_Mites", "Target_Spot", "Yellow_Leaf_Curl_Virus", "Mosaic_Virus", "Healthy"],
    }

    # Use the disease labels for the given plant type.
    class_labels = disease_labels.get(plant_type, ["Unknown"])

    device = torch.device("cuda:0" if torch.cuda.is_available()
                          else "mps:0" if torch.backends.mps.is_available()
    else "cpu")
    model_path = f"src/models/{plant_type}_Disease_Classification.pth"
    # Load the disease classifier using the file path.
    model = load_model(DiseaseClassifier, model_path, num_classes=len(class_labels), device=device)

    # Preprocess the image.
    image_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()

    disease_prediction = class_labels[predicted_idx]
    print(f"Predicted Disease: {disease_prediction} (Index: {predicted_idx})")
    return disease_prediction
