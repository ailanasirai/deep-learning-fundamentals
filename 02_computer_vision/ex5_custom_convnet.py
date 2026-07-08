# ============================================================
# Exercise 5: Custom Convolutional Network
# Course  : Computer Vision (Kaggle)
# Repo    : deep-learning-fundamentals
# Dataset : Car or Truck (binary image classification)
# Architecture: 3-block custom ConvNet from scratch
# ============================================================

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory

# ------------------------------------------------------------
# Reproducibility — same results every run
# ------------------------------------------------------------
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed()

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Load Car or Truck dataset
# image_size=[128,128]: all images resized to same shape
# label_mode='binary': car=0, truck=1
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
# Q1: Custom ConvNet — 3 blocks
#
# Block 1: 32 filters — learns basic features (edges, colors)
# Block 2: 64 filters, 2 Conv layers — learns shapes
# Block 3: 128 filters, 3 Conv layers — learns complex parts
#
# Filter count doubles each block: 32 -> 64 -> 128
# More Conv layers = larger receptive field per block
# padding='same': output size = input size after Conv
# MaxPool2D: halves spatial dimensions after each block
#
# Head:
# Flatten: 3D feature maps -> 1D vector
# Dense(6, relu): combine features
# Dropout(0.2): prevent overfitting
# Dense(1, sigmoid): binary output (car or truck probability)
# ------------------------------------------------------------
model = keras.Sequential([
    # Block One
    layers.Conv2D(filters=32, kernel_size=3, activation='relu',
                  padding='same', input_shape=[128, 128, 3]),
    layers.MaxPool2D(),

    # Block Two
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Block Three
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),

    # Head
    layers.Flatten(),
    layers.Dense(6, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid'),
])

model.summary()

# ------------------------------------------------------------
# Q2: Compile
# binary_crossentropy: correct loss for 0/1 classification
# binary_accuracy: % images correctly classified
# Adam(epsilon=0.01): stable convergence on image data
# ------------------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(epsilon=0.01),
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

# ------------------------------------------------------------
# Train for 50 epochs
# No early stopping here — observe full training behavior
# ------------------------------------------------------------
history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=50,
    verbose=1,
)

# ------------------------------------------------------------
# Plot results
# Compare loss and accuracy curves
# ------------------------------------------------------------
history_frame = pd.DataFrame(history.history)

history_frame.loc[:, ['loss', 'val_loss']].plot()
plt.title("Loss — Custom ConvNet")
plt.savefig('ex5_loss.png', dpi=150)
plt.show()

history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
plt.title("Accuracy — Custom ConvNet")
plt.savefig('ex5_accuracy.png', dpi=150)
plt.show()

print("Best Validation Accuracy: {:.4f}".format(
    history_frame['val_binary_accuracy'].max()))
