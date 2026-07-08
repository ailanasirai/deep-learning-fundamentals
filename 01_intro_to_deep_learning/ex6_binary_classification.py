# ============================================================
# Exercise 6: Binary Classification
# Course  : Intro to Deep Learning (Kaggle)
# Repo    : deep-learning-fundamentals
# Dataset : Hotel Cancellations (predict if booking cancelled)
# ============================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from tensorflow import keras
from tensorflow.keras import layers

# ------------------------------------------------------------
# Load Hotel Cancellations dataset
# Target: is_canceled (0 = stayed, 1 = cancelled)
# stratify=y ensures equal class ratio in train/valid split
# ------------------------------------------------------------
hotel = pd.read_csv('../input/dl-course-data/hotel.csv')
X = hotel.copy()
y = X.pop('is_canceled')

X['arrival_date_month'] = X['arrival_date_month'].map(
    {'January':1, 'February':2, 'March':3, 'April':4,
     'May':5, 'June':6, 'July':7, 'August':8,
     'September':9, 'October':10, 'November':11, 'December':12}
)

features_num = [
    "lead_time", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies",
    "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "required_car_parking_spaces",
    "total_of_special_requests", "adr",
]
features_cat = [
    "hotel", "arrival_date_month", "meal",
    "market_segment", "distribution_channel",
    "reserved_room_type", "deposit_type", "customer_type",
]

transformer_num = make_pipeline(
    SimpleImputer(strategy="constant"),
    StandardScaler(),
)
transformer_cat = make_pipeline(
    SimpleImputer(strategy="constant", fill_value="NA"),
    OneHotEncoder(handle_unknown='ignore'),
)

preprocessor = make_column_transformer(
    (transformer_num, features_num),
    (transformer_cat, features_cat),
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, stratify=y, train_size=0.75
)
X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)
input_shape = [X_train.shape[1]]
print("Input shape:", input_shape)

# ------------------------------------------------------------
# Q1: Model architecture from diagram
# BatchNorm -> Dense -> BatchNorm -> Dropout ->
# Dense -> BatchNorm -> Dropout -> Dense(sigmoid)
# sigmoid output: probability between 0 and 1
# 0.7 = 70% chance booking will be cancelled
# ------------------------------------------------------------
model = keras.Sequential([
    layers.BatchNormalization(input_shape=input_shape),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid'),
])

# ------------------------------------------------------------
# Q2: Compile with binary crossentropy + accuracy
# binary_crossentropy: correct loss for 0/1 classification
# binary_accuracy: what % of predictions are correct
# ------------------------------------------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

# ------------------------------------------------------------
# Train with early stopping
# 200 epochs max but stops when val_loss plateaus
# restore_best_weights: go back to best checkpoint
# ------------------------------------------------------------
early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    min_delta=0.001,
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=200,
    callbacks=[early_stopping],
    verbose=0,
)

history_df = pd.DataFrame(history.history)

history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy Loss")
plt.savefig('ex6_loss.png', dpi=150)
plt.show()

history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")
plt.savefig('ex6_accuracy.png', dpi=150)
plt.show()

print("Training stopped at epoch:", len(history_df))
print("Best Validation Accuracy: {:0.4f}".format(
    history_df['val_binary_accuracy'].max()))
