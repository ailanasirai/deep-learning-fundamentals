# ============================================================
# Exercise 3: Stochastic Gradient Descent
# Course  : Intro to Deep Learning (Kaggle)
# Repo    : deep-learning-fundamentals
# Dataset : Fuel Economy (predicting fuel efficiency)
# ============================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector

# ------------------------------------------------------------
# Load and preprocess data
# Numerical features: StandardScaler (mean=0, std=1)
# Categorical features: OneHotEncoder (text -> 0/1 columns)
# Target: log(FE) instead of raw FE to reduce skew
# ------------------------------------------------------------
fuel = pd.read_csv('../input/dl-course-data/fuel.csv')

X = fuel.copy()
y = X.pop('FE')

preprocessor = make_column_transformer(
    (StandardScaler(),
     make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse=False),
     make_column_selector(dtype_include=object)),
)

X = preprocessor.fit_transform(X)
y = np.log(y)

input_shape = [X.shape[1]]
print("Input shape:", input_shape)
# Output: [50] — 50 features after encoding

# ------------------------------------------------------------
# Define model
# 128 -> 128 -> 64 -> 1
# ReLU on hidden layers, no activation on output (regression)
# ------------------------------------------------------------
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1),
])

# ------------------------------------------------------------
# Q1: Compile — optimizer + loss
# Adam: adjusts learning rate automatically per parameter
# MAE: mean absolute error, robust to outliers
# ------------------------------------------------------------
model.compile(
    optimizer='adam',
    loss='mae'
)

# ------------------------------------------------------------
# Q2: Train — 200 epochs, batch size 128
# batch_size=128: weights updated after every 128 samples
# epochs=200: full dataset seen 200 times
# verbose=0: silent training
# ------------------------------------------------------------
history = model.fit(
    X, y,
    epochs=200,
    batch_size=128,
    verbose=0
)

# ------------------------------------------------------------
# Plot loss curve
# Start at epoch 5 to skip the initial steep drop
# A still-decreasing curve means model can train longer
# ------------------------------------------------------------
history_df = pd.DataFrame(history.history)
history_df.loc[5:, ['loss']].plot()
plt.xlabel("Epoch")
plt.ylabel("MAE Loss")
plt.title("Training Loss — Fuel Economy")
plt.tight_layout()
plt.savefig('ex3_loss_curve.png', dpi=150)
plt.show()

print("Final MAE loss:", round(history_df['loss'].iloc[-1], 4))

# ------------------------------------------------------------
# SGD Animation experiments
# Observe: learning rate controls step size
#          batch size controls update noise
# ------------------------------------------------------------
from learntools.deep_learning_intro.dltools import animate_sgd
plt.rc('animation', html='html5')

# Stable baseline
animate_sgd(learning_rate=0.05, batch_size=32,
            num_examples=256, steps=50, true_w=3.0, true_b=2.0)

# Large learning rate — overshoots
animate_sgd(learning_rate=0.2, batch_size=32,
            num_examples=256, steps=50, true_w=3.0, true_b=2.0)

# Tiny batch — noisy updates
animate_sgd(learning_rate=0.05, batch_size=2,
            num_examples=256, steps=50, true_w=3.0, true_b=2.0)
