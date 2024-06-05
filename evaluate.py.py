import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load model
model_path = '../models/2.keras'
model = tf.keras.models.load_model(model_path)

# Load test dataset
IMAGE_SIZE = 256
BATCH_SIZE = 32
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)
class_names = dataset.class_names

def get_dataset_partitions(ds, train_split=0.8, val_split=0.1, test_split=0.1):
    ds_size = len(ds)
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    test_size = int(test_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size).take(test_size)
    
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_dataset_partitions(dataset)

# Predictions on test set
plt.figure(figsize=(15, 15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_class = class_names[np.argmax(model.predict(np.expand_dims(images[i], axis=0)))]
        actual_class = class_names[labels[i].numpy()] 
        
        plt.title(f"Actual: {actual_class}\nPredicted: {predicted_class}")
        plt.axis("off")
plt.show()
