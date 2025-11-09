import os, json, torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
from collections import Counter

# We have to normalize using ImageNet stats since we are using a model pretrained on ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

DATA_DIR = "PlantVillage"   
MODEL_NAME = "deit_small_patch16_224"
BATCH_SIZE = 8
EPOCHS = 1
LR = 5e-4
WEIGHT_DECAY = 5e-2
LABEL_SMOOTHING = 0.1
OUT_DIR = "debug_output"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

os.makedirs(OUT_DIR, exist_ok=True)

# ------------------- TRANSFORMS -------------------
# We use simple transforms since ViTs are quite robust to image variations
# Resize → CenterCrop → ToTensor → Normalize.
simple_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# ------------------- DATASET & SPLIT -------------------
# Load all data from PlantVillage/train
# ImageFolder expects subfolders for each class: train/<class_name>/*.jpg
full_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=simple_tf)
num_classes = len(full_ds.classes)
print(f"Found {num_classes} classes:", full_ds.classes)

# Split dataset into 80-20 train-val
val_size = int(0.2 * len(full_ds))
train_size = len(full_ds) - val_size
# We split into train and validation sets
train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))

# Dataloaders
train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# Save class-to-index mapping
with open(os.path.join(OUT_DIR, "class_to_idx.json"), "w") as f:
    json.dump(full_ds.class_to_idx, f, indent=2)

# ------------------- MODEL -------------------
# deit_small_patch16_224 is a small Vision Transformer model with 16x16 patches
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes).to(device)

# We freeze everything except the classification head
for p in model.parameters():
    p.requires_grad = False
for p in model.head.parameters():
    p.requires_grad = True

# We have 384 * num_classes + num_classes trainable parameters in the head
# In this case, num_classes = 38
# So, trainable parameters = 384 * 3 + 3 = 1155
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters (head only): {n_trainable}")

# ------------------- CLASS WEIGHTS -------------------
def compute_class_weights(dataset):
    counts = Counter(dataset.dataset.targets)  # get counts from full_ds
    total = sum(counts.values())
    weights = torch.zeros(len(dataset.dataset.classes), dtype=torch.float32)
    for cls_idx in range(len(dataset.dataset.classes)):
        n = counts.get(cls_idx, 1)
        weights[cls_idx] = total / (len(dataset.dataset.classes) * n)
    return weights

class_weights = compute_class_weights(train_ds).to(device)
print("Class weights:", class_weights.tolist())

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

# ------------------- OPTIMIZER -------------------
decay, no_decay = [], []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if p.ndim == 1 or n.endswith("bias") or "norm" in n.lower():
        no_decay.append(p)
    else:
        decay.append(p)

optimizer = torch.optim.AdamW([
    {"params": decay, "weight_decay": WEIGHT_DECAY},
    {"params": no_decay, "weight_decay": 0.0}
], lr=LR)

scaler = torch.cuda.amp.GradScaler()

# ------------------- EVALUATION FUNCTION -------------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.numel()
    return correct / max(1, total)

# ------------------- TRAINING LOOP -------------------
best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for x, y in train_ld:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * y.size(0)

    train_loss = running_loss / len(train_ds)
    val_acc = evaluate(model, val_ld)
    print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_head.pt"))

print(f"Best Validation Accuracy: {best_acc:.4f}")
