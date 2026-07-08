# ============================================================
# Exercise 4: The Sliding Window
# Course  : Computer Vision (Kaggle)
# Repo    : deep-learning-fundamentals
# Concept : Receptive fields, stacked layers, 1D convolution
# ============================================================

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import learntools.computer_vision.visiontools as visiontools
from learntools.computer_vision.visiontools import (
    edge, blur, bottom_sobel, emboss, sharpen, circle
)

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')

# ------------------------------------------------------------
# Load images and kernels
# ------------------------------------------------------------
image_dir = '../input/computer-vision-resources/'
circle_64 = tf.expand_dims(circle([64, 64], val=1.0, r_shrink=4), axis=-1)
kaggle_k = visiontools.read_image(image_dir + 'k.jpg', channels=1)
car = visiontools.read_image(image_dir + 'car_illus.jpg', channels=1)
car = tf.image.resize(car, size=[200, 200])

images = [(circle_64, "circle_64"), (kaggle_k, "kaggle_k"), (car, "car")]
plt.figure(figsize=(14, 4))
for i, (img, title) in enumerate(images):
    plt.subplot(1, len(images), i+1)
    plt.imshow(tf.squeeze(img))
    plt.axis('off')
    plt.title(title)
plt.savefig('ex4_images.png', dpi=150)
plt.show()

kernels_list = [(edge, "edge"), (blur, "blur"),
                (bottom_sobel, "bottom_sobel"),
                (emboss, "emboss"), (sharpen, "sharpen")]
plt.figure(figsize=(14, 4))
for i, (krn, title) in enumerate(kernels_list):
    plt.subplot(1, len(kernels_list), i+1)
    visiontools.show_kernel(krn, digits=2, text_size=20)
    plt.title(title)
plt.savefig('ex4_kernels.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# Feature extraction experiments
# car + edge: highlights car body outlines
# circle + bottom_sobel: detects bottom edges of circle
# kaggle_k + sharpen: enhances letter detail
# ------------------------------------------------------------
experiments = [
    (car, edge, "car + edge"),
    (circle_64, bottom_sobel, "circle + bottom_sobel"),
    (kaggle_k, sharpen, "kaggle_k + sharpen"),
]

for img, krn, name in experiments:
    print(f"\nExperiment: {name}")
    visiontools.show_extraction(
        img, krn,
        conv_stride=1,
        conv_padding='valid',
        pool_size=2,
        pool_stride=2,
        pool_padding='same',
        subplot_shape=(1, 4),
        figsize=(14, 6),
    )
    plt.suptitle(name)
    plt.savefig(f'ex4_{name.replace(" + ", "_")}.png', dpi=150)
    plt.show()

# ------------------------------------------------------------
# Q1: Receptive Field Growth
# Layer 1 (3x3): receptive field = 3x3
# Layer 2 (3x3): receptive field = 5x5
# Layer 3 (3x3): receptive field = 7x7
# Each layer adds 2 to each dimension
# 3 x (3x3) = 27 params vs 1 x (7x7) = 49 params
# Same receptive field, 45% fewer parameters
# ------------------------------------------------------------

# ------------------------------------------------------------
# 1D Convolution on Time Series
# Same concept as 2D but sliding window goes left-right only
# detrend [-1,1]: detects week-to-week CHANGES
# average [0.2x5]: smooths out noise, low frequency trends
# spencer: heavy smoother, removes seasonal fluctuations
# ------------------------------------------------------------
machinelearning = pd.read_csv(
    '../input/computer-vision-resources/machinelearning.csv',
    parse_dates=['Week'],
    index_col='Week',
)
machinelearning.plot()
plt.title("ML Search Popularity — Original")
plt.savefig('ex4_timeseries_original.png', dpi=150)
plt.show()

detrend = tf.constant([-1, 1], dtype=tf.float32)
average = tf.constant([0.2, 0.2, 0.2, 0.2, 0.2], dtype=tf.float32)
spencer = tf.constant(
    [-3, -6, -5, 3, 21, 46, 67, 74, 67, 46, 32, 3, -5, -6, -3],
    dtype=tf.float32
) / 320

for kernel, name in [
    (detrend, 'detrend'),
    (average, 'average'),
    (spencer, 'spencer')
]:
    ts_data = machinelearning.to_numpy()
    ts_data = tf.expand_dims(ts_data, axis=0)
    ts_data = tf.cast(ts_data, dtype=tf.float32)
    kern = tf.reshape(kernel, shape=(*kernel.shape, 1, 1))

    ts_filter = tf.nn.conv1d(
        input=ts_data, filters=kern, stride=1, padding='VALID'
    )

    pd.Series(tf.squeeze(ts_filter).numpy()).plot()
    plt.title(f"Filtered: {name}")
    plt.savefig(f'ex4_timeseries_{name}.png', dpi=150)
    plt.show()
