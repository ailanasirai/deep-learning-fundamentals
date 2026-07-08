# ============================================================
# Exercise 2: Convolution and ReLU
# Course  : Computer Vision (Kaggle)
# Repo    : deep-learning-fundamentals
# Concept : Feature extraction via convolution + ReLU
# ============================================================

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sympy
from IPython.display import display

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
tf.config.run_functions_eagerly(True)

# ------------------------------------------------------------
# Load and display image
# channels=1: grayscale (no RGB needed for feature detection)
# ------------------------------------------------------------
image_path = '../input/computer-vision-resources/car_illus.jpg'
image = tf.io.read_file(image_path)
image = tf.io.decode_jpeg(image, channels=1)
image = tf.image.resize(image, size=[400, 400])

img = tf.squeeze(image).numpy()
plt.figure(figsize=(6, 6))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title("Original Image")
plt.savefig('ex2_original.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# Q1: Define edge detection kernel (3x3)
# Center=8, surrounding=-1
# Flat areas -> output near 0
# Edges -> output high (pixel differs from neighbors)
# ------------------------------------------------------------
kernel = tf.constant([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
])

# Reformat for TensorFlow
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)
kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
kernel = tf.cast(kernel, dtype=tf.float32)

# ------------------------------------------------------------
# Q2: Apply convolution
# tf.nn.conv2d = backend of layers.Conv2D
# Slides kernel over image, multiplies and sums each patch
# padding='SAME': output same size as input
# ------------------------------------------------------------
conv_fn = tf.nn.conv2d

image_filter = conv_fn(
    input=image,
    filters=kernel,
    strides=1,
    padding='SAME',
)
plt.imshow(tf.squeeze(image_filter))
plt.axis('off')
plt.title("After Convolution")
plt.savefig('ex2_convolution.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# Q3: Apply ReLU
# tf.nn.relu = backend of layers.Activation('relu')
# Negatives -> 0, Positives -> unchanged
# Only strong detected features survive
# ------------------------------------------------------------
relu_fn = tf.nn.relu

image_detect = relu_fn(image_filter)
plt.imshow(tf.squeeze(image_detect))
plt.axis('off')
plt.title("After ReLU")
plt.savefig('ex2_relu.png', dpi=150)
plt.show()

# ------------------------------------------------------------
# Q4: Numerical example
# Kernel [[1,-1],[1,-1]] detects vertical edges
# Left pixel - right pixel: high where edge exists
# ReLU removes negatives (edges going wrong direction)
# ------------------------------------------------------------
sympy.init_printing()

image_num = np.array([
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 1],
    [0, 1, 0, 0, 0, 0],
])
kernel_num = np.array([
    [1, -1],
    [1, -1],
])

image_num = tf.cast(image_num, dtype=tf.float32)
image_num = tf.reshape(image_num, [1, *image_num.shape, 1])
kernel_num = tf.reshape(kernel_num, [*kernel_num.shape, 1, 1])
kernel_num = tf.cast(kernel_num, dtype=tf.float32)

image_filter_num = tf.nn.conv2d(
    input=image_num, filters=kernel_num, strides=1, padding='VALID'
)
image_detect_num = tf.nn.relu(image_filter_num)

print("After Convolution:")
display(sympy.Matrix(tf.squeeze(image_filter_num).numpy()))
print("After ReLU:")
display(sympy.Matrix(tf.squeeze(image_detect_num).numpy()))
