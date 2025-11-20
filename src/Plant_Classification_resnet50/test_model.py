import os
import json
import torch
import pandas as pd
from datetime import datetime

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import datasets

from src.inference import classify_plant
from src.inference import PLANT_CLASSES

TEST_DIR = "PlantVillage_Test/valid/"   

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = f"results/plant_classification/run_{timestamp}"
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"\nSaving results to: {RESULTS_DIR}\n")

dataset = datasets.ImageFolder(TEST_DIR)
class_names = dataset.classes
num_classes = len(class_names)

print(f"Found {len(dataset)} test images.")
print("Classes:", class_names)

all_true = []
all_pred = []
all_files = []

for img_path, label in dataset.imgs:
    pred_class = classify_plant(img_path)   
    all_true.append(class_names[label])
    all_pred.append(pred_class)
    all_files.append(img_path)

df = pd.DataFrame({
    "filepath": all_files,
    "true_label": all_true,
    "predicted_label": all_pred,
})
df.to_csv(os.path.join(RESULTS_DIR, "predictions.csv"), index=False)

print("Saved: predictions.csv")

acc = accuracy_score(all_true, all_pred)

print("\nOverall Accuracy:", acc)

print("\nClassification Report:")
report = classification_report(all_true, all_pred, target_names=class_names)
print(report)

with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
    f.write("Accuracy: {:.4f}\n\n".format(acc))
    f.write(report)

print("Saved: classification_report.txt")

cm = confusion_matrix(all_true, all_pred, labels=class_names)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Plant Classification")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.close()

print("Saved: confusion_matrix.png")
print("\nEvaluation complete!\n")
