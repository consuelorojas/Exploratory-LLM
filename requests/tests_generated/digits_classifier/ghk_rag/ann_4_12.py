import tensorflow as tf

import pytest
from your_module import ClassifyDigits, model_path
import numpy as np
from PIL import Image
import os

# Load testing data (MNIST test set)
def load_testing_data():
    # Assuming you have a function to load MNIST dataset or similar single-digit images
    from tensorflow.keras.datasets import mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    
    return x_test, y_test

# Preprocess testing data for the model
def preprocess_testing_data(x):
    # Normalize and flatten input data
    normalized_x = x / 255.0
    flattened_x = normalized_x.reshape(-1, 28 * 28)
    
    return flattened_x

@pytest.fixture
def classify_digits():
    yield ClassifyDigits()

# Test the model with MNIST test set or similar single-digit images
def test_recognize_more_than_95_percent_correctly(classify_digits):
    x_test, y_test = load_testing_data()
    preprocessed_x_test = preprocess_testing_data(x_test)
    
    predictions = classify_digits(preprocessed_x_test)
    accuracy = np.sum(predictions == y_test) / len(y_test)
    
    assert accuracy > 0.95

# Test the model with at least ten different inputs from your dataset
def test_recognize_ten_different_inputs_correctly(classify_digits):
    x_test, _ = load_testing_data()
    preprocessed_x_test = preprocess_testing_data(x_test[:10])
    
    predictions = classify_digits(preprocessed_x_test)
    accuracy = np.sum(predictions == [5, 0, 4, 1, 9, 2, 7, 3, 6, 8]) / len([5, 0, 4, 1, 9, 2, 7, 3, 6, 8])
    
    assert accuracy > 0.95

# Test the model with a single image
def test_recognize_single_image_correctly(classify_digits):
    img_path = 'path_to_your_test_image.png'
    x = np.array(Image.open(img_path).convert('L').resize((28, 28)))
    
    prediction = classify_digits(x)
    assert isinstance(prediction[0], int)

# Test the model with an invalid input
def test_recognize_invalid_input(classify_digits):
    img_path = 'path_to_your_test_image.png'
    x = np.array(Image.open(img_path).convert('L').resize((28, 29)))
    
    with pytest.raises(ValueError) as excinfo:
        classify_digits(x)
