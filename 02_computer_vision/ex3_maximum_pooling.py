# ============================================================
# Exercise 3: Maximum Pooling
# Course  : Computer Vision (Kaggle)
# Repo    : deep-learning-fundamentals
# Concept : Pooling, translation invariance, GlobalAvgPool2D
# ============================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import gridspec
import learntools.computer_vision.visiontools as visiontools
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')

# ------------------------------------------------------------
# Load image + apply emboss kernel (from Ex2)
# Emboss kernel highlights texture and depth differences
# ------------------------------------------------------------
image_path = '../input/computer-vision-resources/car_illus.jpg'
image = tf.io.read_file(image_path)
image = tf.io.decode_jpeg(image, channels=1)
image = tf.image.resize(image, size=[400, 400])

kernel = tf.constant([
    [-2, -1, 0],
    [-1,  1, 1],
    [ 0,  1, 2],
])

image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)
kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
kernel = tf.cast(kernel, dtype=tf.float32)

image_filter = tf.nn.conv2d(
    input=image, filters=kernel, strides=1, padding='VALID'
)
image_detect = tf.nn.relu(image_filter)

plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.imshow(tf.squeeze(image), cmap='gray')
plt.axis('off')
plt.title('Input')
plt.subplot(132)
plt.imshow(tf.squeeze(image_filter))
plt.axis('off')
plt.title('Filter')
plt.subplot(133)
plt.imshow(tf.squeeze(image_detect))
plt.axis('off')
plt.title('Detect')
plt.savefig('ex3_filter_detect.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# Q1: Maximum Pooling
# window_shape=(2,2): look at 2x2 blocks
# pooling_type='MAX': keep strongest activation in each block
# strides=(2,2): move 2 pixels at a time, halves image size
# Result: smaller image, same features, translation robust
# ------------------------------------------------------------
image_condense = tf.nn.pool(
    input=image_detect,
    window_shape=(2, 2),
    pooling_type='MAX',
    strides=(2, 2),
    padding='SAME',
)

plt.figure(figsize=(8, 6))
plt.subplot(121)
plt.imshow(tf.squeeze(image_detect))
plt.axis('off')
plt.title("Detect (ReLU)")
plt.subplot(122)
plt.imshow(tf.squeeze(image_condense))
plt.axis('off')
plt.title("Condense (MaxPool)")
plt.savefig('ex3_maxpool.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# Translation Invariance demo
# Circle randomly shifted, then MaxPool applied 4 times
# After enough pooling: shift doesn't matter, feature same
# ------------------------------------------------------------
REPEATS = 4
SIZE = [64, 64]

image = visiontools.circle(SIZE, r_shrink=4, val=1)
image = tf.expand_dims(image, axis=-1)
image = visiontools.random_transform(image, jitter=3, fill_method='replicate')
image = tf.squeeze(image)

plt.figure(figsize=(16, 4))
plt.subplot(1, REPEATS+1, 1)
plt.imshow(image, vmin=0, vmax=1)
plt.title("Original\nShape: {}x{}".format(image.shape[0], image.shape[1]))
plt.axis('off')

for i in range(REPEATS):
    image = tf.reshape(image, [1, *image.shape, 1])
    image = tf.nn.pool(
        image, window_shape=(2,2), strides=(2,2),
        padding='SAME', pooling_type='MAX'
    )
    image = tf.squeeze(image)
    plt.subplot(1, REPEATS+1, i+2)
    plt.imshow(image, vmin=0, vmax=1)
    plt.title("MaxPool {}\nShape: {}x{}".format(
        i+1, image.shape[0], image.shape[1]))
    plt.axis('off')
plt.savefig('ex3_translation_invariance.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# Global Average Pooling
# 8 feature maps (5x5 each) reduced to 8 single values
# 25x compression, still preserves feature presence info
# Used in head instead of Flatten+Dense for fewer params
# ------------------------------------------------------------
feature_maps = [
    visiontools.random_map([5, 5], scale=0.1, decay_power=4)
    for _ in range(8)
]

gs = gridspec.GridSpec(1, 8, wspace=0.01, hspace=0.01)
plt.figure(figsize=(18, 2))
for i, feature_map in enumerate(feature_maps):
    plt.subplot(gs[i])
    plt.imshow(feature_map, vmin=0, vmax=1)
    plt.axis('off')
plt.suptitle('Feature Maps', size=18, weight='bold', y=1.1)
plt.show()

feature_maps_tf = [
    tf.reshape(fm, [1, *fm.shape, 1]) for fm in feature_maps
]
global_avg_pool = tf.keras.layers.GlobalAvgPool2D()
pooled_maps = [global_avg_pool(fm) for fm in feature_maps_tf]
img = np.array(pooled_maps)[:,:,0].T

plt.imshow(img, vmin=0, vmax=1)
plt.axis('off')
plt.title('Pooled Feature Maps')
plt.savefig('ex3_global_avg_pool.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# VGG16 + GlobalAvgPool2D on Car vs Truck
# 512 feature maps each reduced to 1 value = 512 numbers
# Cars and trucks produce visually different patterns
# Dense(1, sigmoid) can separate them from just 512 values
# ------------------------------------------------------------
pretrained_base = tf.keras.models.load_model(
    '../input/cv-course-models/cv-course-models/vgg16-pretrained-base',
)

model = keras.Sequential([
    pretrained_base,
    layers.GlobalAvgPool2D(),
])

ds = image_dataset_from_directory(
    '../input/car-or-truck/train',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=1,
    shuffle=True,
)
ds_iter = iter(ds)

for _ in range(3):
    car = next(ds_iter)
    car_tf = tf.image.resize(car[0], size=[128, 128])
    car_features = model(car_tf)
    car_features = tf.reshape(car_features, shape=(16, 32))
    label = int(tf.squeeze(car[1]).numpy())

    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.imshow(tf.squeeze(car[0]))
    plt.axis('off')
    plt.title(["Car", "Truck"][label])
    plt.subplot(122)
    plt.imshow(car_features)
    plt.title('Pooled Feature Maps')
    plt.axis('off')
    plt.savefig(f'ex3_car_truck_{_}.png', dpi=150)
    plt.show()
