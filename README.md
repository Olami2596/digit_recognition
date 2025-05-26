# MNIST Digit Recognition using a Simple Neural Network

This repository contains a Jupyter Notebook (`untitled.ipynb`) that demonstrates how to build and train a simple Artificial Neural Network (ANN) for handwritten digit recognition using the widely known MNIST dataset.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Input Image Requirements](#input-image-requirements)
- [Contributing](#contributing)

## Introduction
This project implements a basic neural network to classify handwritten digits (0-9). The MNIST dataset, a classic benchmark in machine learning, is used for training and evaluation. The notebook covers data loading, preprocessing, model definition, training, and evaluation.

## Dataset
The MNIST dataset is a large database of handwritten digits that is commonly used for training various image processing systems. The dataset consists of:

- 60,000 training images and their corresponding labels.
- 10,000 test images and their corresponding labels.
- Each image is a 28x28 pixel grayscale image.

## Model Architecture
The neural network implemented in this notebook is a simple Sequential model from Keras, consisting of:

- A `Flatten` layer to convert the 2D image (28x28) into a 1D array (784 pixels).
- Two `Dense` (fully connected) hidden layers with 50 neurons each, using the `relu` activation function.
- An output `Dense` layer with 10 neurons (one for each digit 0–9), using the `sigmoid` activation function.

The model is compiled with:

- **Optimizer:** `adam`
- **Loss Function:** `sparse_categorical_crossentropy` (suitable for integer labels)
- **Metrics:** `accuracy`

## Results
The model is trained for 10 epochs. After training, the model achieves an accuracy of approximately **97.2%** on the test dataset.

## Dependencies
To run this notebook, you will need the following Python libraries:

- `numpy`
- `matplotlib`
- `seaborn`
- `opencv-python` (`cv2`)
- `Pillow` (`PIL`)
- `tensorflow`
- `keras` (usually installed with tensorflow)

You can install these dependencies using pip:

```bash
pip install numpy matplotlib seaborn opencv-python Pillow tensorflow keras
````

## Usage

Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Open the Jupyter Notebook:

```bash
jupyter notebook untitled.ipynb
```

Run all cells: Execute the cells in the notebook sequentially to load the data, train the model, and evaluate its performance.

## Input Image Requirements

When testing the model with your own images, ensure that the input images are **identical in format and structure** to the ones used in the MNIST dataset. Specifically:

* Images must be 28x28 pixels.
* They should be in **grayscale**.
* The digit should be **centered** and **clearly visible**.
* The background should be black (or dark), and the digit should be white (or light).

Using images that deviate from this structure may result in inaccurate predictions or processing errors.

### Examples:

Refer to the `pictures/` directory in this repository for sample input images that follow the required format.

## Contributing

Feel free to fork this repository, open issues, or submit pull requests. Any contributions are welcome!


