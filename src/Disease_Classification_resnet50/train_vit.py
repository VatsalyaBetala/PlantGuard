import os, json, argparse, torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from collections import Counter

IMAGENET_MEAN=(0.485,0.456,0.406)
IMAGENET_STD=(0.229,0.224,0.225)

def make_loaders(data_dir, batch_size=16, workers=2):
    
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7,1.0)),  # random crop/scale
        transforms.RandomHorizontalFlip(),                   # leaves mirrored
        transforms.ColorJitter(0.2,0.2,0.2,0.05),            # lighting changes
        transforms.RandomRotation(15),                       # tilt
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    # Collects all image paths in a list along with their class IDs.
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf) # apply the trasformation
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=val_tf)

    # Laad the dataset using DataLoader class
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_ds, val_ds, train_ld, val_ld

# ---- evaluation + class weights ----
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)

def compute_class_weights(train_ds):
    # imbalance fix: inverse frequency per class index
    counts = Counter(train_ds.targets)       
    num_classes = len(train_ds.classes)
    total = sum(counts.values())
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for cls_idx in range(num_classes):
        n = counts.get(cls_idx, 1)
        weights[cls_idx] = total / (num_classes * n)
    return weights

# ---- main() with head-only transfer learning ----
def main():
    # Build command-line interfaces (CLI)
    ap = argparse.ArgumentParser() # parser object to which we can add arguments
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--model", type=str, default="deit_small_patch16_224")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-2)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--out", type=str, default="checkpoints")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # data
    train_ds, val_ds, train_ld, val_ld = make_loaders(args.data_dir, args.batch_size)
    num_classes = len(train_ds.classes)
    print("Classes:", train_ds.classes)

    with open(os.path.join(args.out, "class_to_idx.json"), "w") as f:
        json.dump(train_ds.class_to_idx, f, indent=2)

    # model (pretrained) + swap head
    model = timm.create_model(args.model, pretrained=True, num_classes=num_classes).to(device)

    # 3) freeze backbone, train head only
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True

    # count trainable params (should be just the head)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters (head only):", n_trainable)

    # loss with class weights
    class_w = compute_class_weights(train_ds).to(device)
    print("Class weights:", class_w.tolist())
    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=args.label_smoothing)

    # 5) optimizer: AdamW with no weight decay on norm/bias
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith("bias") or 'norm' in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr
    )
    scaler = torch.cuda.amp.GradScaler()

    # 6) training loop
    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * y.size(0)

        train_loss = running / len(train_ds)
        val_acc = evaluate(model, val_ld, device)
        print(f"[Head] Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), os.path.join(args.out, "best_head.pt"))

    print("Best val acc:", round(best, 4))

if __name__ == "__main__":
    main()
