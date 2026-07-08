# ============================================================
# Exercise 4: Overfitting and Underfitting
# Course  : Intro to Deep Learning (Kaggle)
# Repo    : deep-learning-fundamentals
# Dataset : Spotify (predicting track popularity)
# ============================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupShuffleSplit
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# ------------------------------------------------------------
# Load and preprocess Spotify dataset
# Target: track_popularity scaled to 0-1
# Grouped split: all songs of one artist stay in same split
# Prevents signal leakage between train and validation
# ------------------------------------------------------------
spotify = pd.read_csv('../input/dl-course-data/spotify.csv')

X = spotify.copy().dropna()
y = X.pop('track_popularity')
artists = X['track_artist']

features_num = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms']
features_cat = ['playlist_genre']

preprocessor = make_column_transformer(
    (StandardScaler(), features_num),
    (OneHotEncoder(), features_cat),
)

def group_split(X, y, group, train_size=0.75):
    splitter = GroupShuffleSplit(train_size=train_size)
    train, test = next(splitter.split(X, y, groups=group))
    return (X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test])

X_train, X_valid, y_train, y_valid = group_split(X, y, artists)
X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)
y_train = y_train / 100
y_valid = y_valid / 100

input_shape = [X_train.shape[1]]
print("Input shape:", input_shape)

# ------------------------------------------------------------
# Baseline: Linear model (single Dense unit)
# Result: UNDERFITTING — too simple to learn music patterns
# Both train and val loss are high, no gap between them
# ------------------------------------------------------------
model = keras.Sequential([
    layers.Dense(1, input_shape=input_shape),
])
model.compile(optimizer='adam', loss='mae')
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512, epochs=50, verbose=0,
)
history_df = pd.DataFrame(history.history)
history_df.loc[10:, ['loss', 'val_loss']].plot()
plt.title("Linear Model — Underfitting")
plt.savefig('ex4_underfitting.png', dpi=150)
plt.show()
print("Linear Model — Min Val Loss: {:0.4f}".format(history_df['val_loss'].min()))

# ------------------------------------------------------------
# Deeper model: Dense(128) + Dense(64)
# Result: OVERFITTING — training loss drops, val loss rises
# Model memorizes training data, fails on new data
# ------------------------------------------------------------
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mae')
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512, epochs=50, verbose=0,
)
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
plt.title("Deep Model — Overfitting")
plt.savefig('ex4_overfitting.png', dpi=150)
plt.show()
print("Deep Model — Min Val Loss: {:0.4f}".format(history_df['val_loss'].min()))

# ------------------------------------------------------------
# Q3: Early Stopping callback
# patience=5: stop if no improvement for 5 epochs
# min_delta=0.001: improvement must exceed this threshold
# restore_best_weights=True: revert to best checkpoint
# ------------------------------------------------------------
early_stopping = callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)

# ------------------------------------------------------------
# Same deep model + early stopping
# Result: training stops before overfitting begins
# Best weights automatically restored
# ------------------------------------------------------------
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mae')
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512, epochs=50,
    callbacks=[early_stopping],
    verbose=0,
)
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
plt.title("Early Stopping — Just Right")
plt.savefig('ex4_early_stopping.png', dpi=150)
plt.show()
print("Early Stopping — Min Val Loss: {:0.4f}".format(history_df['val_loss'].min()))
print("Training stopped at epoch:", len(history_df))
