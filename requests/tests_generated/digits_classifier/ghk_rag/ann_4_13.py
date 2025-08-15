
import pytest
from your_module import ClassifyDigits, model_path
import numpy as np
from PIL import Image
import os
import tensorflow as tf

# Load test data (MNIST)
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

def create_test_image(image_array):
    """Create a 28x28 image from an array."""
    img = Image.fromarray(np.uint8(255 * image_array))
    return np.array(img)

@pytest.fixture
def classify_digits():
    yield ClassifyDigits()

# Test with MNIST dataset
def test_recognize_more_than_95_percent(classify_digits):
    # Select 10 random images from the training set
    indices = np.random.choice(len(x_train), size=100, replace=False)
    x_test = [create_test_image(image) for image in x_train[indices]]
    
    correct_count = 0
    
    for i, img in enumerate(x_test):
        prediction = classify_digits(np.array([img]))
        if y_train[indices[i]] == int(prediction[0]):
            correct_count += 1
            
    accuracy = (correct_count / len(indices)) * 100
    assert accuracy > 95

# Test with a single image
def test_recognize_single_image(classify_digits):
    # Create an array for the digit '7'
    img_array = np.zeros((28, 28))
    
    # Draw a simple '7' shape on the array
    for i in range(10):
        img_array[5 + i][13] = 255
    
    prediction = classify_digits(np.array([img_array]))
    assert int(prediction[0]) == 7

# Test with an invalid image (not a digit)
def test_recognize_invalid_image(classify_digits, tmp_path):
    # Create an array for the letter 'A'
    img_array = np.zeros((28, 28))
    
    # Draw a simple 'A' shape on the array
    for i in range(10):
        if i < 5:
            img_array[13 - i][i + 3] = 255
        else:
            img_array[i - 2][17 - (i % 6)] = 255
    
    prediction = classify_digits(np.array([img_array]))
    
    # The model should not recognize this as a digit, so the confidence will be low for all digits.
    assert np.max(prediction) < 0.5

# Test with an empty image
def test_recognize_empty_image(classify_digits):
    img_array = np.zeros((28, 28))
    
    prediction = classify_digits(np.array([img_array]))
    
    # The model should not recognize this as a digit.
    assert int(prediction[0]) == -1 or np.max(prediction) < 0.5
