import tensorflow as tf

import pytest
from your_module import ClassifyDigits, model_path
import numpy as np
from PIL import Image
import os

# Load MNIST test data for testing purposes
def load_mnist_test_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return x_test, y_test

@pytest.fixture
def classifier():
    """Create a ClassifyDigits instance"""
    return ClassifyDigits()

@pytest.mark.parametrize("image_path", [
    "path/to/image1.png",
    "path/to/image2.png",
    # Add more image paths here...
])
def test_recognize_digits(classifier, image_path):
    """
    Test the classifier with an individual image
    """
    x = PIL.Image.open(image_path).convert('L').resize((28, 28))
    images = np.array(x)
    prediction = classifier(images=images)
    # Assuming you have a way to get the actual digit from the image path
    expected_digit = int(os.path.basename(image_path)[0])
    assert prediction == expected_digit

def test_recognize_digits_accuracy(classifier):
    """
    Test the accuracy of the classifier with MNIST test data
    """
    x_test, y_test = load_mnist_test_data()
    correct_count = 0
    for i in range(10): # Testing first 10 images from each class (total 100)
        image = x_test[i]
        expected_digit = y_test[i]
        prediction = classifier(images=np.array([image]))
        if prediction == [expected_digit]:
            correct_count += 1

    accuracy = correct_count / 10
    assert accuracy > 0.95, f"Accuracy {accuracy} is less than the required threshold of 0.95"

def test_recognize_digits_multiple_inputs(classifier):
    """
    Test the classifier with multiple inputs from MNIST dataset
    """
    x_test, y_test = load_mnist_test_data()
    correct_count = 0

    # Select at least ten different images for testing (e.g., one image per class)
    test_images = []
    expected_digits = []
    for i in range(10):
        index = np.where(y_test == i)[0][0]
        test_image = x_test[index].reshape((1, 28, 28))
        test_images.append(test_image)
        expected_digits.extend([i])

    # Stack the images into a single array
    test_images_array = np.stack(test_images)

    predictions = classifier(images=test_images_array.reshape(-1, 784))

    for i in range(len(predictions)):
        if int(predictions[i]) == expected_digits[i]:
            correct_count += 1

    accuracy = correct_count / len(expected_digits)
    assert accuracy > 0.95
