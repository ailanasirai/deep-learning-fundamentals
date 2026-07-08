# ============================================================
# Exercise 1: Transfer Learning with InceptionV1
# Course  : Computer Vision (Kaggle)
# Repo    : deep-learning-fundamentals
# Dataset : Car or Truck (binary image classification)
# ============================================================

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Reproducibility
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed()

warnings.filterwarnings("ignore")
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')

# ------------------------------------------------------------
# Load image dataset
# image_size=[128,128]: resize all images to same size
# label_mode='binary': car=0, truck=1
# batch_size=64: process 64 images at once
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

# Convert pixel values to float32 (0-255 to 0.0-1.0)
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = ds_train_.map(convert_to_float).cache().prefetch(AUTOTUNE)
ds_valid = ds_valid_.map(convert_to_float).cache().prefetch(AUTOTUNE)

# ------------------------------------------------------------
# Load pretrained InceptionV1 base
# Already trained on ImageNet (millions of images)
# We use it purely for feature extraction
# ------------------------------------------------------------
import tensorflow_hub as hub
pretrained_base = tf.keras.models.load_model(
    '../input/cv-course-models/cv-course-models/inceptionv1'
)

# ------------------------------------------------------------
# Q1: Freeze the base
# trainable=False: base weights stay fixed during training
# We don't want to overwrite ImageNet features
# Only the head we attach will be trained
# ------------------------------------------------------------
pretrained_base.trainable = False

# ------------------------------------------------------------
# Q2: Attach classification head
# Flatten: converts 3D feature maps to 1D vector
# Dense(6, relu): combines extracted features
# Dense(1, sigmoid): outputs probability (car=0, truck=1)
# ------------------------------------------------------------
model = keras.Sequential([
    pretrained_base,
    layers.Flatten(),
    layers.Dense(6, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])

# ------------------------------------------------------------
# Q3: Compile
# binary_crossentropy: correct loss for 0/1 problems
# binary_accuracy: what % images classified correctly
# Adam with epsilon=0.01: stable optimization for image tasks
# ------------------------------------------------------------
optimizer = tf.keras.optimizers.Adam(epsilon=0.01)
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

model.summary()

# ------------------------------------------------------------
# Train for 30 epochs
# Base is frozen so only Dense layers update
# Fast training because base does heavy feature work
# ------------------------------------------------------------
history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=30,
)

# Plot results
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
plt.title("Loss — InceptionV1 Transfer Learning")
plt.savefig('ex1_loss.png', dpi=150)
plt.show()

history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
plt.title("Accuracy — InceptionV1 Transfer Learning")
plt.savefig('ex1_accuracy.png', dpi=150)
plt.show()

print("Best Val Accuracy: {:0.4f}".format(
    history_frame['val_binary_accuracy'].max()))
