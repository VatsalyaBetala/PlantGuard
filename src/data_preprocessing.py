import os
import zipfile
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import kagglehub
from PIL import Image



def setup_kaggle():

    # Download latest version
    path = kagglehub.dataset_download("emmarex/plantdisease")
    print("Path to dataset files:", path)

def get_transforms():
    return transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor
def get_datasets(transform):
    dataset = datasets.ImageFolder(r"C:\Users\Vatsalya Betala\OneDrive\Documents\Repositories\plant-disease-classification\PlantVillage", transform=transform)
    return dataset

def split_indices(dataset, split, validation_split):
    indices = list(range(len(dataset)))
    split = int(np.floor(split * len(dataset)))
    validation = int(np.floor(validation_split * split))
    np.random.shuffle(indices)
    return indices[:validation], indices[validation:split], indices[split:]

if __name__ == '__main__':
    setup_kaggle()