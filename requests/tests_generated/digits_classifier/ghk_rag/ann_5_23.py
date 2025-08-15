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
    """Returns an instance of the ClassifyDigits class."""
    return ClassifyDigits()

@pytest.mark.parametrize("image_path", [
    "path/to/image1.png",
    "path/to/image2.png",
    # Add more image paths here...
])
def test_recognize_digits(classifier, image_path):
    """
    Test that the classifier recognizes digits correctly.

    Args:
        classifier (ClassifyDigits): An instance of ClassifyDigits.
        image_path (str): Path to an image file containing a digit.
    """
    # Load and preprocess the image
    img = Image.open(image_path).convert('L').resize((28, 28))
    images = np.array(img)

    # Make predictions using the classifier
    prediction = classifier(images=images)
    
    # Assuming we have ground truth labels for our test data...
    expected_label = int(os.path.basename(image_path)[0])  # Replace with actual label

    assert prediction == expected_label, f"Expected {expected_label}, but got {prediction}"

def test_recognize_digits_accuracy(classifier):
    """
    Test that the classifier recognizes over 95% of digits correctly.

    Args:
        classifier (ClassifyDigits): An instance of ClassifyDigits.
    """
    # Load MNIST test data
    x_test, y_test = load_mnist_test_data()

    correct_count = 0

    for i in range(10):
        img = Image.fromarray(x_test[i]).convert('L').resize((28, 28))
        images = np.array(img)

        prediction = classifier(images=images)
        
        if prediction == y_test[i]:
            correct_count += 1
    
    accuracy = (correct_count / len(y_test[:10])) * 100

    assert accuracy > 95.0, f"Expected accuracy above 95%, but got {accuracy:.2f}%"

def test_load_model():
    """
    Test that the model is loaded correctly.
    """
    # Try to load the model
    try:
        tf.keras.models.load_model(model_path)
    except Exception as e:
        pytest.fail(f"Failed to load model: {str(e)}")
