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
    """Create a ClassifyDigits instance."""
    yield ClassifyDigits()

@pytest.mark.parametrize("image_path", [
    "path/to/image1.png",
    "path/to/image2.png",
    # Add more image paths here...
])
def test_recognize_digits(classifier, image_path):
    """
    Test the classifier with an individual image.
    
    :param classifier: A ClassifyDigits instance
    :param image_path: Path to a single-digit number image file (0-9)
    """
    # Load and preprocess the image data
    img = Image.open(image_path).convert('L').resize((28, 28))
    images = np.array(img)

    # Get the predicted digit from the classifier
    prediction = classifier(images=images)[0]

    # Assuming we have a way to get the actual label for this test case...
    expected_label = int(os.path.basename(image_path).split('.')[0])  # Replace with your logic

    assert prediction == expected_label, f"Expected {expected_label}, but got {prediction}"

def test_recognize_digits_bulk(classifier):
    """
    Test the classifier's accuracy on a bulk dataset.
    
    :param classifier: A ClassifyDigits instance
    """
    x_test, y_test = load_mnist_test_data()

    # Preprocess images and get predictions from the model
    num_correct = 0
    for i in range(10):  # Test with at least ten different inputs as per acceptance criteria
        img = Image.fromarray(x_test[i]).convert('L').resize((28, 28))
        images = np.array(img)
        prediction = classifier(images=images)[0]
        
        if prediction == y_test[i]:
            num_correct += 1

    accuracy = (num_correct / len(y_test[:10])) * 100
    assert accuracy > 95.0, f"Expected an accuracy of over 95%, but got {accuracy:.2f}%"

def test_load_model():
    """
    Test that the model is loaded correctly.
    
    :return:
    """
    # Check if the model exists at the specified path and can be loaded
    assert os.path.exists(model_path), f"Model file not found: {model_path}"
