# Plant Disease Classification

This project is focused on classifying plant diseases using a convolutional neural network (CNN) implemented with TensorFlow and Keras. The dataset used is from PlantVillage.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [License](#license)
- [Contributing](#contributing)

## Installation

To install the necessary dependencies, follow these steps:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/VatsalyaBetala/plant-disease-classification.git
    cd plant-disease-classification
    ```

2. **Create a virtual environment** (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The dataset used in this project is from [PlantVillage](https://www.plantvillage.org/). Make sure to download the dataset and place it in the `PlantVillage` directory within the project.

## Usage

### Training the Model

To train the model, run the following command:

```bash
python train.py
