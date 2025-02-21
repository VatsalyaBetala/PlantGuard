import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from src.data_preprocessing import preprocess_image  # Assumes this is your preprocessing function
from src.Plant_Classification_resnet50.plant_classification_TL import PlantClassifierCNN
from src.Disease_Classification_resnet50.src.disease_model import DiseaseClassifier

# Define your plant and disease class labels.
PLANT_CLASSES = ["Apple", "Corn_(maize)", "Grape", "Pepper_bell", "Potato", "Tomato"]


def load_model(model_class, model_path, num_classes, device):
    print(f"Loading model: {model_class.__name__}")
    print(f"Model path: {model_path}")
    print(f"Using device: {device}")

    model = model_class(num_classes)  # Ensure correct initialization
    print("Model initialized!")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)  # Secure loading
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
        raise  # Raise the error so the traceback appears in logs

def classify_plant(image_path: str, model_path: str="src/models/Plant_Classification.pth") -> str:
    """
    Classify a plant image using the PlantClassifierCNN model.
    Loads model weights from the specified file path.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # Load the plant classifier using the file path.
    import os
    print(os.path.exists("src/models/Plant_Classification.pth"))  # Should return True
    model = load_model(PlantClassifierCNN, model_path, num_classes=len(PLANT_CLASSES), device=device)
    print("Hi")
    # Preprocess the image.
    image_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        probabilities = torch.softmax(output, dim=1)  # Get probability distribution
        predicted_idx = torch.argmax(probabilities, dim=1).item()  # Get class index
        predicted_prob = probabilities[0, predicted_idx].item()  # Get probability of predicted class

        print(f"Predicted Class: {predicted_idx}, Probability: {predicted_prob}")

    plant_name = PLANT_CLASSES[predicted_idx]
    if predicted_prob < 0.95:
        print(f"Low confidence prediction: {predicted_prob}")
        plant_name = "Unknown"
    print(f"Predicted Plant: {plant_name} (Index: {predicted_idx})")
    return plant_name


def classify_disease(image_path: str, plant_type: str) -> str:
    """
    Classify the disease on a plant image using the DiseaseClassifier model.
    Loads model weights from the specified file path.
    """
    if plant_type=="Unknown":
        return "Unknown"
    disease_labels = {
        "Apple": ["Apple Scab", "Apple Black Rot", "Cedar Apple Rust", "Apple Healthy"],
        "Corn_(maize)": ["Cercospora_leaf_spot Gray_leaf_spot", "Common Rush", "Northern_Leaf_Blight", "Healthy"],
        "Grape": ["Black Rot", "Esca_(Black_Measles)", "Leaf_blight_(Isariopsis_Leaf_Spot)", "Healthy"],
        "Pepper_bell": ["Bacterial_Spot", "Healthy"],
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
        probabilities = torch.softmax(output, dim=1)  # Get probability distribution
        predicted_idx = torch.argmax(probabilities, dim=1).item()  # Get class index
        predicted_prob = probabilities[0, predicted_idx].item()  # Get probability of predicted class

        print(f"Predicted Class: {predicted_idx}, Probability: {predicted_prob}")

    disease_prediction = class_labels[predicted_idx]
    if predicted_prob < 0.95:
        print(f"Low confidence prediction: {predicted_prob}")
        disease_prediction = "Unknown"
    print(f"Predicted Plant: {disease_prediction} (Index: {predicted_idx})")
    return disease_prediction
