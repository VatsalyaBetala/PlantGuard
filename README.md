# **🌱 Plant Disease Classification**
An AI-powered **deep learning model** for detecting and classifying **plant diseases** using **Convolutional Neural Networks (CNNs)** with **TensorFlow and Keras**. This project aims to provide **farmers and agricultural professionals** with a tool for **early disease detection**, reducing crop losses and improving food security.

---

## **📑 Table of Contents**
- [Introduction](#introduction)
- [Team](#team)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Run the Web UI](#run-the-web-ui)
  - [Switch Models](#switch-models)
  - [Artifacts Layout](#artifacts-layout)
  - [Training the Model](#training-the-model)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Deployment](#deployment)
- [License](#license)
- [Contributing](#contributing)

---

## **Introduction**
Plant diseases pose a **significant threat to global food security**, causing **20-40% yield losses annually** (*FAO, 2021*). Current diagnostic methods are **slow, expensive, and require expert intervention**.  
This project offers an **AI-based, real-time, offline disease detection tool**, enabling **farmers, researchers, and agribusinesses** to:
- **Detect diseases early**, preventing crop loss.
- **Classify multiple plant diseases** with high accuracy.
- **Deploy AI models on IoT devices** like **Raspberry Pi** for offline use.

---

## **Team**
| Name | Role | Background |
|------|------|------------|
| **Vatsalya Betala** | AI/ML Lead | BSc. (Hons) **Computer Science & Math** |
| **Aarav Akali** | Hardware & IoT Lead | BSc. (Hons) **Computer Science Major, Econ Minor, Physics Concentration** |
| **Sparsh Makharia** | Backend Developer & Business Lead | BSc. (Hons) **Computer Science & Data Science with Business Concentration** |

We are a **multidisciplinary team**, combining **AI, IoT, and business expertise** to bring **cutting-edge technology to agriculture**.

---

## **Installation**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/VatsalyaBetala/plant-disease-classification.git
cd plant-disease-classification
```

### **2️⃣ Create a Virtual Environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # Python >= 3.8, on Windows, use `venv\Scripts\activate`
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## **Dataset**
This project uses the **PlantVillage dataset**, which contains **thousands of labeled images** of healthy and diseased plant leaves.  
- Download the dataset from [PlantVillage](https://www.plantvillage.org/).
- Place it in the `PlantVillage/` directory within the project.

---

## **Usage**
### **Run the Web UI**
```bash
python main.py
```
The FastAPI UI will be available at `http://127.0.0.1:8000`.

### **Switch Models**
Select a backend using an environment variable (default is the CNN backend):
```bash
MODEL_NAME=cnn_resnet50 python main.py
```

### **Artifacts Layout**
Place model artifacts here so the UI can load them uniformly:
```
artifacts/
  <model_name>/
    model.pt
    labels.json
    config.yaml (optional)
  shared/
    resnet50_weights.pth
    yolov8n_leaf.pt
```

### **Training the Model**
To train the model, run:
```bash
python train/train.py
```
This will:
- Load the dataset.
- Train a CNN model.
- Save the trained model in the current directory.

---

## **Model Architecture**
The model is built using **Transfer Learning with ResNet-50**, which provides:
- **Pretrained feature extraction** from **millions of images**.
- **Fine-tuned classification layers** for detecting **plant diseases**.
- **High inference efficiency** for deployment on **low-power edge devices (Raspberry Pi, Jetson Nano).**

---

## **Results**
Our model achieves:
- **87%+ accuracy** in classifying multiple plant diseases.
- **Fast inference speed (~50ms per image)** on a **GPU**.
- **Optimized for edge deployment** with **model quantization**.

---

## **Web Interface**
PlantGuard comes with a lightweight web UI built with **FastAPI** and **Materialize CSS**. Features include:
- Image uploads with live previews and progress indicator.
- A gallery view with on-demand AI explanations and image download option.
- Dark mode support with persistent preference.
- Informative **About** and **Contact** pages.

---

## **Deployment**
### **1️⃣ Running Inference Locally**
Run the UI and upload an image via the web interface:
```bash
python main.py
```

### **2️⃣ Deploying on Raspberry Pi**
Follow the same UI workflow and point to artifacts on the device filesystem.

---

## **License**
This project is **open-source** and licensed under the **MIT License**.

---

## **Contributing**
Want to improve the project?  
- **Fork the repository**.
- **Create a branch** (`git checkout -b feature-name`).
- **Submit a pull request** (`PR`).

For major changes, open an **issue** first to discuss proposed modifications.
---
