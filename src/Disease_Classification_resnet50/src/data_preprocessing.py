import os
from torchvision import datasets, transforms

def get_transforms():
    """Define transformations applied to images for data augmentation."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_datasets(base_path):
    """Load datasets for a specific plant."""
    train_path = os.path.join(base_path, 'train')
    valid_path = os.path.join(base_path, 'valid')
    
    transform = get_transforms()

    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    valid_dataset = datasets.ImageFolder(valid_path, transform=transform)

    return train_dataset, valid_dataset
