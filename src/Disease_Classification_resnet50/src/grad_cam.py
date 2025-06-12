import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from disease_model import DiseaseClassifier

IMG_PATH = r'C:\Users\Vatsalya Betala\OneDrive\Documents\Repositories\plant-disease-classification\src\Disease_Classification_resnet50\src\Late_Blight_1.jpg'
MODEL_PATH = r'C:\Users\Vatsalya Betala\OneDrive\Documents\Repositories\plant-disease-classification\src\models\Potato_Disease_Classification.pth'
NUM_CLASSES = 3  


device = torch.device("cpu")

model = DiseaseClassifier(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval() # Prediction Mode

gradients = []
activations = []

def save_activation(module, input, output):
    activations.append(output)
    output.register_hook(lambda grad: gradients.append(grad))

target_layer = model.model.layer4[-1]
target_layer.register_forward_hook(save_activation)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

img = Image.open(IMG_PATH).convert('RGB')
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

cam = np.maximum(cam, 0) # ReLU
cam = cv2.resize(cam, (224, 224)) # Resizing
cam -= cam.min() # Normalizng
cam /= cam.max() 

# Overlay
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
img_np = np.array(img.resize((224, 224)))
overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

plt.imshow(overlay)
plt.title(f'Grad-CAM: Class {class_idx}')
plt.axis('off')
plt.show()
