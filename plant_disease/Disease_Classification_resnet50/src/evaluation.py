import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from disease_model import DiseaseClassifier 

def load_model(model_path, num_classes, device):
    """Load the trained model."""
    model = DiseaseClassifier(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_validation_data(base_path):
    """Load validation dataset."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_path = f"{base_path}/valid"
    dataset = datasets.ImageFolder(valid_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    return dataset, dataloader

def evaluate_model(model, dataloader, device):
    """Evaluate the model on the validation set."""
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    return accuracy, avg_loss

if __name__ == "__main__":
    
    plant_name = "Apple"  # Change for each plant (e.g., Grape, Tomato)
    model_path = f"{plant_name}_Disease_Classification.pth"
    base_path = f"/data_home/vatsalya/PlantDiseaseClassification/Disease_Classification/data/Apple"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model for {plant_name}...")
    dataset, dataloader = get_validation_data(base_path)
    model = load_model(model_path, num_classes=len(dataset.classes), device=device)

    print(f"Evaluating model on {plant_name} validation set...")
    accuracy, loss = evaluate_model(model, dataloader, device)

    print(f"Model: {plant_name}_Disease_Classification.pth")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Validation Loss: {loss:.4f}")
