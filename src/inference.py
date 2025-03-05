import os
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO  # (kept for any other parts; not used in detect_leaf)
import onnxruntime as ort  # For running the ONNX model
from torchvision.ops import nms
from src.utils import check_and_download_models
from src.data_preprocessing import preprocess_image
from src.Plant_Classification_resnet50.plant_classification_TL import PlantClassifierCNN
from src.Disease_Classification_resnet50.src.disease_model import DiseaseClassifier


# ─────────────────────────────────────────────────────────────────────────────
# 1) Leaf detection function using ONNX
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
import os
import cv2
import torch
import numpy as np
from PIL import Image
import onnxruntime as ort
from torchvision.ops import nms  # Using torchvision's non-max suppression


# ─────────────────────────────────────────────────────────────────────────────
# 1) Leaf detection function using ONNX with calibration adjustments
# ─────────────────────────────────────────────────────────────────────────────
def detect_leaf(image_path: str):
    """
    Detects a leaf using the YOLOv8n ONNX model with output shape [1, 5, 8400].
    Returns the path to the cropped leaf image ('temp_leaf.jpg'), or None if detection fails.
    """
    # Load ONNX model
    onnx_model_path = "src/models/yolov8n_leaf.onnx"
    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape  # Expected: [1, 3, 640, 640]
    input_height, input_width = input_shape[2:4]

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return None

    # Letterbox preprocessing
    def letterbox(im, new_shape=(640, 640)):
        shape = im.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        im = cv2.copyMakeBorder(im, int(dh), int(dh), int(dw), int(dw),
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return im, r, (dw, dh)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img, ratio, (dw, dh) = letterbox(img)
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img).astype(np.float32) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Run inference
    outputs = session.run(None, {input_name: img})
    outputs = torch.tensor(outputs[0])  # Shape: [1, 5, 8400]

    # Decode outputs
    def decode_yolo_output(outputs, conf_thres=0.8, iou_thres=0.5):
        # Reshape and filter outputs
        outputs = outputs.squeeze(0)  # Remove batch dimension
        outputs = outputs.permute(1, 0)  # [8400, 5]

        # Split into boxes and confidence scores
        boxes = outputs[:, :4]  # [x_center, y_center, width, height]
        scores = outputs[:, 4]  # Confidence scores

        # Convert boxes from [x_center, y_center, width, height] to [x1, y1, x2, y2]
        boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2)  # x1
        boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2)  # y1
        boxes[:, 2] = (boxes[:, 0] + boxes[:, 2])  # x2
        boxes[:, 3] = (boxes[:, 1] + boxes[:, 3])  # y2

        # Apply confidence threshold
        mask = scores > conf_thres
        boxes = boxes[mask]
        scores = scores[mask]

        # Apply NMS
        keep_indices = nms(boxes, scores, iou_thres)
        return boxes[keep_indices], scores[keep_indices]

    # Decode and filter outputs
    boxes, scores = decode_yolo_output(outputs)

    if boxes.shape[0] == 0:
        print("No detections.")
        return None

    # Use the first detection
    box = boxes[0].numpy()
    x1, y1, x2, y2 = box

    # Scale coordinates back to the original image size
    x1 = (x1 - dw) / ratio
    y1 = (y1 - dh) / ratio
    x2 = (x2 - dw) / ratio
    y2 = (y2 - dh) / ratio

    # Clip coordinates to the image dimensions
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(image.shape[1], int(x2))
    y2 = min(image.shape[0], int(y2))

    # Crop the detected leaf and save the image
    leaf_crop = image[y1:y2, x1:x2]
    if leaf_crop.size == 0:
        print("Warning: Cropped leaf is empty.")
        return None
    cropped_leaf = Image.fromarray(cv2.cvtColor(leaf_crop, cv2.COLOR_BGR2RGB))
    cropped_leaf_path = "temp_leaf.jpg"
    cropped_leaf.save(cropped_leaf_path)
    print("Leaf detected and cropped! with Confidence: ", scores[0].item())
    return cropped_leaf_path
# ─────────────────────────────────────────────────────────────────────────────
# 2) Helper to load a PyTorch model
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_class, model_path, num_classes, device):
    print(f"Loading model: {model_class.__name__}")
    print(f"Model path: {model_path}")
    print(f"Using device: {device}")

    model = model_class(num_classes)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        print("Checkpoint loaded successfully!")

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        print("Model loaded and ready!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 3) Plant classification (expects a cropped leaf image!)
# ─────────────────────────────────────────────────────────────────────────────
PLANT_CLASSES = ["Apple", "Corn_(maize)", "Grape", "Pepper_bell", "Potato", "Tomato"]

def classify_plant(cropped_leaf_path: str, model_path: str = "src/models/Plant_Classification.pth") -> str:
    """
    Classify a cropped leaf image using the PlantClassifierCNN model.
    """
    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Load the plant classifier
    model = load_model(
        model_class=PlantClassifierCNN,
        model_path=model_path,
        num_classes=len(PLANT_CLASSES),
        device=device
    )

    # Preprocess the cropped leaf image
    image_tensor = preprocess_image(cropped_leaf_path).to(device)

    # Inference
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        predicted_prob = probabilities[0, predicted_idx].item()

    plant_name = PLANT_CLASSES[predicted_idx]
    if predicted_prob < 0.50:
        print(f"Low confidence prediction: {predicted_prob:.2f}")
        plant_name = "Unknown"

    print(f"Predicted Plant: {plant_name} (Confidence: {predicted_prob:.2f})")
    return plant_name


# ─────────────────────────────────────────────────────────────────────────────
# 4) Disease classification (expects a cropped leaf image!)
# ─────────────────────────────────────────────────────────────────────────────
def classify_disease(cropped_leaf_path: str, plant_type: str) -> str:
    """
    Classify the disease on a cropped leaf image using the DiseaseClassifier model.
    """
    if plant_type == "Unknown":
        return "Unknown"

    disease_labels = {
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
    class_labels = disease_labels.get(plant_type, ["Unknown"])  # default to ["Unknown"] if not found

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Dynamically determine model file based on plant name
    model_path = f"src/models/{plant_type}_Disease_Classification.pth"
    if not os.path.exists(model_path):
        print(f"No disease model found for {plant_type}, returning 'Unknown'")
        return "Unknown"

    # Load disease classifier
    model = load_model(
        model_class=DiseaseClassifier,
        model_path=model_path,
        num_classes=len(class_labels),
        device=device
    )

    # Preprocess the cropped leaf image
    image_tensor = preprocess_image(cropped_leaf_path).to(device)

    # Inference
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        predicted_prob = probabilities[0, predicted_idx].item()

    disease_prediction = class_labels[predicted_idx]
    if predicted_prob < 0.50:
        print(f"Low confidence disease prediction: {predicted_prob:.2f}")
        disease_prediction = "Unknown"

    print(f"Predicted Disease: {disease_prediction} (Confidence: {predicted_prob:.2f})")
    return disease_prediction