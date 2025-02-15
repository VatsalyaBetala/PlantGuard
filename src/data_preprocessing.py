import os
import zipfile
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import kagglehub


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