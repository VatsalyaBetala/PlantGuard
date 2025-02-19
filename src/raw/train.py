import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
from raw.plant_classification import PlantClassifierCNN
from raw.data_preprocessing import get_datasets, get_transforms

def train_model(train_loader, validation_loader, device, epochs, model, criterion, optimizer, scheduler):
    train_losses = []  
    validation_losses = [] 

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(validation_loader)
        validation_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        scheduler.step()

    return train_losses, validation_losses


def main():
    base_path = "/data_home/vatsalya/PlantDiseaseClassification/Plant_Classification/data"
    train_dataset, valid_dataset = get_datasets(base_path)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PlantClassifierCNN(len(train_dataset.classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.02)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_losses, validation_losses = train_model(train_loader, valid_loader, device, 10, model, criterion, optimizer, scheduler)

    torch.save(model.state_dict(), 'plant_disease_model.pth')
    print("Model saved to 'plant_disease_model.pth'.")
    print("Training and validation completed successfully.")

if __name__ == '__main__':
    main()


