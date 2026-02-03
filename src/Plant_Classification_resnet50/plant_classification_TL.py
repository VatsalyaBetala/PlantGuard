import torch
import torch.nn as nn
import torchvision.models as models

from src.model_artifacts import resolve_shared_file

class PlantClassifierCNN(nn.Module):
    def __init__(self, num_classes = 2):
        super(PlantClassifierCNN, self).__init__()

        # Load ResNet50 without downloading
        self.model = models.resnet50(weights=None)
        weight_path = resolve_shared_file("resnet50_weights.pth")
        # Load manually downloaded weights
        self.model.load_state_dict(torch.load(weight_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

        # Freeze all layers except the last classification layer
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the final classification layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)
    
