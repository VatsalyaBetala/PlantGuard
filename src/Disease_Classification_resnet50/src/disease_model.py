import torch.nn as nn
import torchvision.models as models
import torch

class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DiseaseClassifier, self).__init__()

        # Load ResNet50 and freeze weights
        self.model = models.resnet50(weights=None)
        weight_path = "src/models/resnet50_weights.pth"
        self.model.load_state_dict(torch.load(weight_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

        for param in self.model.parameters():
            param.requires_grad = False  # Freeze all layers except the last one
        
        # Unfreeze the last conv block (layer4) for Grad-CAM
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # Number of disease classes for the plant
        )

    def forward(self, x):
        return self.model(x)
