import os
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.Disease_Classification_resnet50.src.disease_model import DiseaseClassifier

# Mapping plant type to disease labels (same as in inference.py)
DISEASE_LABELS = {
    "Apple": ["Apple Scab", "Apple Black Rot", "Cedar Apple Rust", "Apple Healthy"],
    "Corn_(maize)": ["Cercospora_leaf_spot Gray_leaf_spot", "Common Rush", "Northern_Leaf_Blight", "Healthy"],
    "Grape": ["Black Rot", "Esca_(Black_Measles)", "Leaf_blight_(Isariopsis_Leaf_Spot)", "Healthy"],
    "Pepper_bell": ["Bacterial_Spot", "Healthy"],
    "Potato": ["Early_Blight", "Late_Blight", "Healthy"],
    "Tomato": [
        "Bacterial_Spot", "Early_Blight", "Late_Blight", "Leaf_Mold",
        "Septoria_leaf_spot", "Spider_Mites", "Target_Spot",
        "Yellow_Leaf_Curl_Virus", "Mosaic_Virus", "Healthy"
    ],
}

def generate_grad_cam(image_path: str, plant_type: str) -> Optional[str]:
    """Generate a Grad-CAM heatmap overlay for the given cropped leaf image.

    Parameters
    ----------
    image_path : str
        Path to the cropped leaf image.
    plant_type : str
        Plant prediction used to select the correct disease model.

    Returns
    -------
    Optional[str]
        Path to the saved heatmap image, or ``None`` if generation failed.
    """
    if plant_type not in DISEASE_LABELS:
        return None

    model_path = os.path.join("src", "models", f"{plant_type}_Disease_Classification.pth")
    if not os.path.exists(model_path):
        return None

    num_classes = len(DISEASE_LABELS[plant_type])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load disease classification model
    model = DiseaseClassifier(num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    gradients = []
    activations = []

    def save_activation(module, inp, out):
        activations.append(out)
        out.register_hook(lambda grad: gradients.append(grad))

    handle = model.model.layer4[-1].register_forward_hook(save_activation)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    output = model(input_tensor)
    class_idx = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, class_idx].backward()

    activation = activations[0].squeeze(0).detach().cpu().numpy()
    gradient = gradients[0].squeeze(0).detach().cpu().numpy()
    weights = gradient.mean(axis=(1, 2))

    cam = np.zeros(activation.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activation[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, img.size)
    cam -= cam.min()
    if cam.max() != 0:
        cam /= cam.max()

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_np = np.array(img)
    overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

    os.makedirs("heatmaps", exist_ok=True)
    output_path = os.path.join("heatmaps", os.path.basename(image_path))
    Image.fromarray(overlay).save(output_path)

    handle.remove()
    return output_path
