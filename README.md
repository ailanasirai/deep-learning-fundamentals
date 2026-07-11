# deep-learning-fundamentals

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

> Neural networks built from scratch — not copied, not skimmed. Every exercise understood, implemented, and documented.

Two complete Kaggle courses. Twelve exercises. Real datasets. Loss curves that actually converge.

---

## Courses Completed

| Course | Exercises | Certificate |
|--------|-----------|-------------|
| [Intro to Deep Learning](https://www.kaggle.com/learn/intro-to-deep-learning) | 6 | ✅ Earned |
| [Computer Vision](https://www.kaggle.com/learn/computer-vision) | 6 | ✅ Earned |

---

## 01 — Intro to Deep Learning

Starting point: linear units. Ending point: a binary classifier trained on hotel bookings with BatchNorm, Dropout, and EarlyStopping all working together. Six exercises that built on each other rather than existing in isolation.

### Exercise Breakdown

| # | File | Dataset | Problem | Approach |
|---|------|---------|---------|----------|
| 1 | `ex1_single_neuron.py` | Red Wine Quality (1,599 samples) | Predict wine quality score from 11 chemical measurements | Single `Dense(1)` layer — mathematically identical to linear regression. Inspected raw weights before training. |
| 2 | `ex2_deep_neural_networks.py` | Concrete Compressive Strength (1,030 samples) | Predict concrete strength from 8 ingredients | 3 hidden layers, 512 units each, ReLU activation. Explored separating activation into its own layer for BatchNorm compatibility. |
| 3 | `ex3_stochastic_gradient_descent.py` | Fuel Economy (car features) | Predict fuel efficiency from engine type, year, and specs | Adam optimizer, MAE loss, 200 epochs, batch size 128. SGD animation showed how learning rate and batch size change everything. |
| 4 | `ex4_overfitting_and_underfitting.py` | Spotify Track Popularity | Predict song popularity from audio features | Three models compared: linear (underfitting), deep (overfitting), deep + EarlyStopping (just right). patience=5, min_delta=0.001. |
| 5 | `ex5_dropout_and_batch_normalization.py` | Spotify + Concrete | Fix overfitting and unstable training | Dropout(0.3) on Spotify reduced val_loss gap. BatchNorm on raw Concrete data turned a blank loss curve into a converging one. |
| 6 | `ex6_binary_classification.py` | Hotel Cancellations | Predict if a booking will be cancelled (0 or 1) | sigmoid output, binary_crossentropy loss, binary_accuracy metric. Full pipeline: BatchNorm + Dropout + EarlyStopping. |

### Key Insights

- A loss curve still dropping at epoch 200 means the model had not finished learning — training just stopped
- `Dropout(0.3)` and `BatchNormalization` solve different problems. One fights overfitting. One fights unstable gradients on unscaled data
- `binary_crossentropy` penalizes confident wrong predictions harder than uncertain ones — that asymmetry matters
- Adam adjusts learning rate per parameter automatically. That is why it outperforms plain SGD without manual tuning

---

## 02 — Computer Vision

Moving from numbers to images changed how I think about what a neural network actually does. These six exercises went from understanding individual pixel operations to building and training a custom ConvNet from scratch.

### Exercise Breakdown

| # | File | Dataset | Problem | Approach |
|---|------|---------|---------|----------|
| 1 | `ex1_transfer_learning.py` | Car or Truck (5,117 train images) | Binary image classification | InceptionV1 pretrained on ImageNet, trainable=False. Head: Flatten + Dense(6, relu) + Dense(1, sigmoid). 30 epochs. |
| 2 | `ex2_convolution_and_relu.py` | Car illustration (single image) | Understand feature extraction | Edge detection kernel [-1,-1,-1 / -1,8,-1 / -1,-1,-1]. tf.nn.conv2d then tf.nn.relu. Numerical matrix showed vertical edge detection as actual numbers. |
| 3 | `ex3_maximum_pooling.py` | Car illustration + Car or Truck | Understand pooling and invariance | MaxPool 2x2, stride 2. Translation invariance demo: circle shifted randomly, result identical after 4x pooling. GlobalAvgPool2D reduced 512 feature maps to 512 values. |
| 4 | `ex4_sliding_window.py` | Car illustration + ML Google Trends (5 years) | Receptive fields and 1D convolution | Three stacked 3x3 kernels = 7x7 receptive field with 27 params vs 49 for one 7x7. tf.nn.conv1d applied detrend and smoothing kernels to time series. |
| 5 | `ex5_custom_convnet.py` | Car or Truck | Build ConvNet from scratch | 3-block architecture: Conv2D(32) → Conv2D(64,64) → Conv2D(128,128,128). Filters double each block. 50 epochs. Competitive with pretrained model. |
| 6 | `ex6_data_augmentation.py` | Car or Truck | Reduce overfitting with augmentation | RandomContrast(0.10) + RandomFlip(horizontal) + RandomRotation(0.10) built inside model. BatchNorm(renorm=True) before each block. Filters: 64, 128, 256. |

### Key Insights

- A CNN does not see a car. It sees numbers. Edges are just pixels that differ from their neighbors
- Three stacked 3x3 kernels have 27 parameters and see a 7x7 patch. One 7x7 kernel has 49 parameters and sees the same patch. Depth wins on efficiency
- Translation invariance is real: randomly shift a circle, apply MaxPool four times, the result looks identical every run
- Augmentation inside the model means training=True activates it, inference does not. No separate pipeline needed
- A blank loss curve on unstandardized data is not a mystery — it is SGD failing because the scale differences make gradient updates meaningless

---

## Stack

```
Python 3.10
TensorFlow 2.x
Keras (tf.keras)
NumPy
Pandas
Matplotlib
scikit-learn
```

---

## Repository Structure

```
deep-learning-fundamentals/
│
├── 01_intro_to_deep_learning/
│   ├── ex1_single_neuron.py
│   ├── ex2_deep_neural_networks.py
│   ├── ex3_stochastic_gradient_descent.py
│   ├── ex4_overfitting_and_underfitting.py
│   ├── ex5_dropout_and_batch_normalization.py
│   └── ex6_binary_classification.py
│
├── 02_computer_vision/
│   ├── ex1_transfer_learning.py
│   ├── ex2_convolution_and_relu.py
│   ├── ex3_maximum_pooling.py
│   ├── ex4_sliding_window.py
│   ├── ex5_custom_convnet.py
│   └── ex6_data_augmentation.py
│
├── LICENSE
└── README.md
```

---

## Connect

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/aila-nasir)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://kaggle.com/ailanasirai)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ailanasirai)
