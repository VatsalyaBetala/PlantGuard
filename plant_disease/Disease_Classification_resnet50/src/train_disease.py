import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from disease_model import DiseaseClassifier
from data_preprocessing import get_datasets
import os

def train_model(model, criterion, optimizer, train_loader, valid_loader, device, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(valid_loader)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return model

def main(plant_name):
    base_path = f"/data_home/vatsalya/PlantDiseaseClassification/Disease_Classification/data/{plant_name}"
    train_dataset, valid_dataset = get_datasets(base_path)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    num_classes = len(train_dataset.classes)

    print(f"Training Disease Classification Model for {plant_name}")
    print(f"Number of disease classes: {train_dataset.classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    model = DiseaseClassifier(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.model.fc.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    trained_model = train_model(model, criterion, optimizer, train_loader, valid_loader, device, epochs=10)

    model_path = f"{plant_name}_Disease_Classification.pth"
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved to '{model_path}'.")
    print(f"Training for {plant_name} completed successfully.")
    print()
    print()

if __name__ == '__main__':
    plants = ["Apple", "Corn_(maize)", "Grape", "Pepper,_bell", "Potato", "Tomato"]
    for plant in plants:
        main(plant)
