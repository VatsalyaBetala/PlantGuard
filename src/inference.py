import os
import torch
from PIL import Image
from src.data_preprocessing import preprocess_image
from src.Plant_Classification_resnet50.plant_classification_TL import PlantClassifierCNN
from src.Disease_Classification_resnet50.src.disease_model import DiseaseClassifier


def classify_plant(image_path: str) -> str:
    """
    Classify a plant image using the PlantClassifierCNN model.

    Returns the predicted plant class if the raw logit of the predicted class
    is above the threshold. Otherwise, returns a low confidence message.

    Note: The threshold is in the same scale as the model's logits.
    """
    class_labels = ["Apple", "Corn_(maize)", "Grape", "Pepper,_bell", "Potato", "Tomato"]
    device = torch.device("cuda:0" if torch.cuda.is_available()
                          else "mps:0" if torch.backends.mps.is_available()
    else "cpu")

    model = PlantClassifierCNN(num_classes=len(class_labels))
    model.to(device)
    model.eval()

    image_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        output = model(image_tensor)  # raw logits, shape [1, num_classes]
        # Use torch.argmax to get the predicted index.
        predicted_idx = torch.argmax(output, dim=1).item()
    print(predicted_idx)
    return class_labels[predicted_idx]

def classify_disease(image_path: str, plant_type: str) -> str:
    """
    Classify the disease on a plant image using the DiseaseClassifier model.

    Returns the predicted disease if the raw logit of the predicted class
    is above the threshold. Otherwise, returns a low confidence message.

    Note: The threshold is in the same scale as the model's logits.
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

    class_labels = disease_labels.get(plant_type, ["Unknown"])
    device = torch.device("cuda:0" if torch.cuda.is_available()
                          else "mps:0" if torch.backends.mps.is_available()
    else "cpu")

    model = DiseaseClassifier(num_classes=len(class_labels))
    model.to(device)
    model.eval()

    image_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        # Get the raw logit value for that class.
    return class_labels[predicted_idx]