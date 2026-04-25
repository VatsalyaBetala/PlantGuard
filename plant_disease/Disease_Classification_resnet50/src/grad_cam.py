import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from disease_model import DiseaseClassifier



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
