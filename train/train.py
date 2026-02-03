import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim

from train.model import CNN
from src.data_preprocessing import get_datasets, get_transforms, split_indices, setup_kaggle


def train_model(dataset, train_sampler, validation_sampler, device, epochs, model, criterion, optimizer):
    train_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=64, sampler=validation_sampler)

    train_losses = np.zeros(epochs)
    validation_losses = np.zeros(epochs)

    for epoch in range(epochs):
        model.train()
        train_loss = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            print("Working")
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        train_losses[epoch] = np.mean(train_loss)

        model.eval()
        validation_loss = []
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                validation_loss.append(loss.item())

        validation_losses[epoch] = np.mean(validation_loss)

        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[epoch]:.4f}, "
            f"Validation Loss: {validation_losses[epoch]:.4f}"
        )

    return train_losses, validation_losses


def main():
    # setup_kaggle()  # Set up Kaggle and download the dataset if not already present
    transform = get_transforms()
    dataset = get_datasets(transform)
    train_idx, val_idx, _ = split_indices(dataset, 0.85, 0.70)

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(val_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(len(dataset.classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train_losses, validation_losses = train_model(
        dataset, train_sampler, validation_sampler, device, 10, model, criterion, optimizer
    )

    # Saving the model
    torch.save(model.state_dict(), "plant_disease_model.pth")
    print("Model saved to plant_disease_model.pth.")


if __name__ == '__main__':
    main()
