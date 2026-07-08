# ============================================================
# Exercise 2: Deep Neural Networks
# Course  : Intro to Deep Learning (Kaggle)
# Repo    : deep-learning-fundamentals
# Dataset : Concrete Compressive Strength (1,030 samples, 8 features)
# ============================================================

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
concrete = pd.read_csv('../input/dl-course-data/concrete.csv')
print("Dataset shape:", concrete.shape)
print(concrete.head())

# ------------------------------------------------------------
# Q1: Input shape
# 9 total columns - 1 target (CompressiveStrength) = 8 features
# ------------------------------------------------------------
input_shape = [8]

# ------------------------------------------------------------
# Q2: Deep model with 3 hidden layers
# Each layer has 512 units and ReLU activation
# Output layer has 1 unit, no activation (regression task)
# ReLU: max(0, x) — removes negatives, lets network learn non-linear patterns
# Without activation, stacking Dense layers = still just one linear equation
# ------------------------------------------------------------
model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=[8]),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])

model.summary()

# ------------------------------------------------------------
# Q3: Activation as a separate layer
# Equivalent to Dense(32, activation='relu')
# But separating it lets you insert BatchNorm between Dense and Activation
# ------------------------------------------------------------
model = keras.Sequential([
    layers.Dense(32, input_shape=[8]),
    layers.Activation('relu'),
    layers.Dense(32),
    layers.Activation('relu'),
    layers.Dense(1),
])

# ------------------------------------------------------------
# Optional: Compare activation functions visually
# relu, elu, selu, swish all behave differently below zero
# ------------------------------------------------------------
for act in ['relu', 'elu', 'selu', 'swish']:
    activation_layer = layers.Activation(act)
    x = tf.linspace(-3.0, 3.0, 100)
    y = activation_layer(x)
    plt.figure(dpi=100)
    plt.plot(x, y)
    plt.title(act.upper())
    plt.xlim(-3, 3)
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.tight_layout()
    plt.savefig(f'ex2_activation_{act}.png', dpi=150)
    plt.show()

print("Done.")
