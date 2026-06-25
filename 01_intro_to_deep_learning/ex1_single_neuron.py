# ============================================================
# Exercise 1: A Single Neuron
# Course  : Intro to Deep Learning (Kaggle)
# Repo    : dl-projects-core
# Dataset : Red Wine Quality (1,599 samples, 11 features)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
red_wine = pd.read_csv('../input/dl-course-data/red-wine.csv')

print("Dataset shape:", red_wine.shape)
# Output: (1599, 12)
# 12 columns = 11 features + 1 target (quality)

print(red_wine.head())

# ------------------------------------------------------------
# Q1: Input shape
# 12 total columns - 1 target column = 11 features
# input_shape tells Keras how many values each sample has
# ------------------------------------------------------------
input_shape = [11]
print("Input shape:", input_shape)

# ------------------------------------------------------------
# Q2: Define the linear model
# Dense(units=1) = single neuron = one output value
# input_shape=[11] = accepts 11 feature values per sample
# This is mathematically identical to linear regression:
# output = w1*x1 + w2*x2 + ... + w11*x11 + b
# ------------------------------------------------------------
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[11])
])

model.summary()

# ------------------------------------------------------------
# Q3: Inspect weights before training
# model.weights returns [weight_matrix, bias_vector]
# At this point weights are randomly initialized
# Training will adjust them to fit the data
# ------------------------------------------------------------
w, b = model.weights
print("\nWeights (randomly initialized):\n", w.numpy())
print("\nBias (randomly initialized):\n", b.numpy())

# ------------------------------------------------------------
# Optional: Plot predictions of untrained model
# Result: a random line (weights are random, model has no knowledge yet)
# This is what gradient descent will eventually fix
# ------------------------------------------------------------
model_1d = keras.Sequential([
    layers.Dense(1, input_shape=[1]),
])

x = tf.linspace(-1.0, 1.0, 100)
y = model_1d.predict(x, verbose=0)

w_plot, b_plot = model_1d.weights

plt.style.use('seaborn-whitegrid')
plt.figure(dpi=100)
plt.plot(x, y, 'k')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("Input: x")
plt.ylabel("Predicted output: y")
plt.title("Untrained model\nWeight: {:.2f}  Bias: {:.2f}".format(
    w_plot[0][0].numpy(), b_plot[0].numpy()
))
plt.tight_layout()
plt.savefig('ex1_untrained_linear_model.png', dpi=150)
plt.show()

print("\nDone. Weights are random until training begins.")
