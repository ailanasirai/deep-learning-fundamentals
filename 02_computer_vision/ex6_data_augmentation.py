# ============================================================
# Exercise 6: Data Augmentation
# Course  : Computer Vision (Kaggle)
# Repo    : deep-learning-fundamentals
# Dataset : Car or Truck (binary image classification)
# Concept : Augmentation inside model, BatchNorm, deeper ConvNet
# ============================================================

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, preprocessing
from tensorflow.keras.preprocessing import image_dataset_from_directory

def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed()

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
ds_train_ = image_dataset_from_directory(
    '../input/car-or-truck/train',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=True,
)
ds_valid_ = image_dataset_from_directory(
    '../input/car-or-truck/valid',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=False,
)

def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = ds_train_.map(convert_to_float).cache().prefetch(AUTOTUNE)
ds_valid = ds_valid_.map(convert_to_float).cache().prefetch(AUTOTUNE)

# ------------------------------------------------------------
# Augmentation preview — see what transformations do
# Only active during training=True
# ------------------------------------------------------------
augment_preview = keras.Sequential([
    preprocessing.RandomFlip(mode='horizontal'),
    preprocessing.RandomContrast(factor=0.5),
    preprocessing.RandomRotation(factor=0.10),
])

ex = next(iter(ds_train.unbatch().map(lambda x, y: x).batch(1)))
plt.figure(figsize=(10, 10))
for i in range(16):
    image = augment_preview(ex, training=True)
    plt.subplot(4, 4, i+1)
    plt.imshow(tf.squeeze(image))
    plt.axis('off')
plt.suptitle("Augmentation Preview", size=18)
plt.savefig('ex6_augmentation_preview.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# Q3: Model with augmentation layers built in
#
# Augmentation:
# RandomContrast(0.10): vary lighting by 10%
# RandomFlip(horizontal): mirror left-right (valid for cars)
# RandomRotation(0.10): tilt up to 10% = ~36 degrees
#
# Architecture changes from Ex5:
# Filters: 64, 128, 256 (larger capacity)
# BatchNormalization(renorm=True): stable with augmented batches
# No Dropout in head (BatchNorm handles regularization)
# Dense(8) instead of Dense(6): slightly more capacity
# ------------------------------------------------------------
model = keras.Sequential([
    layers.InputLayer(input_shape=[128, 128, 3]),

    # Data Augmentation (only active during training)
    preprocessing.RandomContrast(factor=0.10),
    preprocessing.RandomFlip(mode='horizontal'),
    preprocessing.RandomRotation(factor=0.10),

    # Block One
    layers.BatchNormalization(renorm=True),
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Block Two
    layers.BatchNormalization(renorm=True),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Block Three
    layers.BatchNormalization(renorm=True),
    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Head
    layers.BatchNormalization(renorm=True),
    layers.Flatten(),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])

model.summary()

# ------------------------------------------------------------
# Compile and train
# ------------------------------------------------------------
optimizer = tf.keras.optimizers.Adam(epsilon=0.01)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=50,
    verbose=1,
)

history_frame = pd.DataFrame(history.history)

history_frame.loc[:, ['loss', 'val_loss']].plot()
plt.title("Loss — Data Augmentation ConvNet")
plt.savefig('ex6_loss.png', dpi=150)
plt.show()

history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
plt.title("Accuracy — Data Augmentation ConvNet")
plt.savefig('ex6_accuracy.png', dpi=150)
plt.show()

print("Best Validation Accuracy: {:.4f}".format(
    history_frame['val_binary_accuracy'].max()))
