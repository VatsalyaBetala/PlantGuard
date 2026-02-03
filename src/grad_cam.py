import os
from typing import Optional
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.Disease_Classification_resnet50.src.disease_model import DiseaseClassifier
from src.model_artifacts import resolve_model_path
from src.model_catalog import DISEASE_LABELS, LEGACY_DISEASE_MODEL_FILENAMES, disease_model_name

# The underlying understanding of how GradCams works lies the understanding of the gradients:

# NOTE:GradCams looks at the last convolution layer, just before the fully-connected layer(s) (the last Conv-layer is unfreezed to capture gradients).

# STEP1: Firstly, the feature maps are calculated by convolving each filter (in this case 2048) over the output of the previous layer.

# STEP2: This leaves you with this activations of dimension : [1,2048,7,7]. (2048 filters, each outputting 7x7 feature map)

# STEP3: Then, for each filter map is AveragePooled, which leaves you with one value for each feature map: [1,2048].

# STEP4: We calculte the weighted sum of feature maps. Following is the dimensional understanding:
# activation[i]      |  [7, 7]  | The i-th feature map A^i from layer4      |
# w                  |  scalar  | a(i), the importance of that feature map  |
# w * activation[i]  |  [7, 7]  | Scaled feature map                        |
# cam += ...         |  [7, 7]  | Running sum of all weighted feature maps  |

# RESULT: A [1,7,7], which serves as the GRAD-CAM heatmap, where each pixel tells you how important that spatial location was for the predicted class

# POSTPROCESSING: Finally, we apply ReLU, resize it, and normalize it.

# We apply a visual overlay on the original image.


def generate_grad_cam(image_path: str, plant_type: str) -> Optional[str]:
    if plant_type not in DISEASE_LABELS:
        return None

    legacy_filename = LEGACY_DISEASE_MODEL_FILENAMES.get(plant_type)
    model_path = resolve_model_path(disease_model_name(plant_type), legacy_filename)
    if not os.path.exists(model_path):
        return None

    num_classes = len(DISEASE_LABELS[plant_type])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DiseaseClassifier(num_classes)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()  # Eval mode - for inference

    gradients = []
    activations = []

    def save_activation(module, inp, out):
        activations.append(out)
        out.register_hook(lambda grad: gradients.append(grad))

    handle = model.model.layer4[-1].register_forward_hook(save_activation)  # Layer-4 of the CNN (ResNet50)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Model Inference (forward pass)
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
