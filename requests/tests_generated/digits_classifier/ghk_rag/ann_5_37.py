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
    """Fixture to create a ClassifyDigits instance"""
    yield ClassifyDigits()

@pytest.mark.parametrize("image_path", [
    os.path.join(os.getcwd(), "test_images/0.png"),
    os.path.join(os.getcwd(), "test_images/1.png"),
    os.path.join(os.getcwd(), "test_images/2.png"),
    # Add more test images here
])
def test_recognize_digits(classifier, image_path):
    """Test recognizing digits with a single input"""
    x = Image.open(image_path).convert('L').resize((28, 28))
    images = np.array(x)
    prediction = classifier(images=images)
    # Assuming the correct label is in the filename
    expected_label = int(os.path.basename(image_path)[0])
    assert prediction == [expected_label]

def test_recognize_digits_bulk(classifier):
    """Test recognizing digits with multiple inputs"""
    x_test, y_test = load_mnist_test_data()
    num_correct = 0
    for i in range(10):  # Test at least ten different inputs
        image = x_test[i]
        images = np.array(image)
        prediction = classifier(images=images)
        if prediction == [y_test[i]]:
            num_correct += 1

    accuracy = (num_correct / 10) * 100
    assert accuracy > 95, f"Accuracy {accuracy} is less than expected"

def test_recognize_non_digits(classifier):
    """Test recognizing non-digit inputs"""
    # Create a random image that doesn't resemble any digit
    x = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
    images = np.array(x)

    prediction = classifier(images=images)
    assert not (prediction >= 0 and prediction <= 9) or len(prediction) == 1

def test_invalid_input(classifier):
    """Test handling invalid input"""
    with pytest.raises(ValueError):
        # Try passing an empty array
        images = np.array([])
        classifier(images=images)

# Add more tests as needed to cover different scenarios
