# ============================================================
# Exercise 5: Dropout and Batch Normalization
# Course  : Intro to Deep Learning (Kaggle)
# Repo    : deep-learning-fundamentals
# Datasets: Spotify (dropout) + Concrete (batch normalization)
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

# ============================================================
# PART 1: DROPOUT — Spotify Dataset
# ============================================================

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

# ------------------------------------------------------------
# Q1: Add Dropout(0.3) after Dense(128) and Dense(64)
# 30% of neurons randomly zeroed each training step
# Forces model to learn redundant representations
# Reduces overfitting seen in Exercise 4
# ------------------------------------------------------------
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
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
plt.title("Dropout Model — Spotify")
plt.savefig('ex5_dropout.png', dpi=150)
plt.show()
print("Dropout Model — Min Val Loss: {:0.4f}".format(history_df['val_loss'].min()))

# ============================================================
# PART 2: BATCH NORMALIZATION — Concrete Dataset (unstandardized)
# ============================================================

concrete = pd.read_csv('../input/dl-course-data/concrete.csv')
df = concrete.copy()
df_train = df.sample(frac=0.7, random_state=0)
df_valid = df.drop(df_train.index)

X_train = df_train.drop('CompressiveStrength', axis=1)
X_valid = df_valid.drop('CompressiveStrength', axis=1)
y_train = df_train['CompressiveStrength']
y_valid = df_valid['CompressiveStrength']
input_shape = [X_train.shape[1]]

# ------------------------------------------------------------
# Without BatchNorm: SGD fails on unstandardized data
# Loss becomes NaN or explodes — blank graph expected
# ------------------------------------------------------------
model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=input_shape),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])
model.compile(optimizer='sgd', loss='mae', metrics=['mae'])
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=64, epochs=100, verbose=0,
)
history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['loss', 'val_loss']].plot()
plt.title("Without BatchNorm — Training Fails")
plt.savefig('ex5_no_batchnorm.png', dpi=150)
plt.show()
print("No BatchNorm — Min Val Loss: {:0.4f}".format(history_df['val_loss'].min()))

# ------------------------------------------------------------
# Q3: Add BatchNormalization before each Dense layer
# Normalizes inputs: mean=0, std=1 per batch
# SGD can now handle unscaled raw data
# ------------------------------------------------------------
model = keras.Sequential([
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu', input_shape=input_shape),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(1),
])
model.compile(optimizer='sgd', loss='mae', metrics=['mae'])
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=64, epochs=100, verbose=0,
)
history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['loss', 'val_loss']].plot()
plt.title("With BatchNorm — Training Works")
plt.savefig('ex5_with_batchnorm.png', dpi=150)
plt.show()
print("With BatchNorm — Min Val Loss: {:0.4f}".format(history_df['val_loss'].min()))
