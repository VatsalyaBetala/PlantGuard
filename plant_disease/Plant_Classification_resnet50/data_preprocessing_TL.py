import os
from torchvision import datasets, transforms

def get_transforms():
    return transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

def get_datasets(base_path):
    transform = get_transforms()
    train_path = os.path.join(base_path, 'train')
    valid_path = os.path.join(base_path, 'valid')

    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    valid_dataset = datasets.ImageFolder(valid_path, transform=transform)

    return train_dataset, valid_dataset
