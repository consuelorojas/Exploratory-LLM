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
    "path_to_image_0.png",
    "path_to_image_1.png",
    # Add more image paths here...
])
def test_recognize_digits(classifier, image_path):
    """Test recognizing digits with a single input."""
    x = np.array(Image.open(image_path).convert('L').resize((28, 28)))
    prediction = classifier(images=x)
    assert isinstance(prediction[0], int)

def test_accuracy_with_mnist_data(classifier):
    """Test the accuracy of digit recognition using MNIST data."""
    # Load MNIST test data
    x_test, y_test = load_mnist_test_data()
    
    correct_predictions = 0
    
    for i in range(10):  # Test with at least ten different inputs
        image = x_test[i]
        prediction = classifier(images=np.array([image]))
        
        if prediction[0] == y_test[i]:
            correct_predictions += 1
            
    accuracy = (correct_predictions / len(x_test[:10])) * 100
    
    assert accuracy > 95

def test_recognize_multiple_digits(classifier):
    """Test recognizing multiple digits."""
    # Load MNIST test data
    x_test, _ = load_mnist_test_data()
    
    images = np.array([x_test[0], x_test[1]])
    predictions = classifier(images=images)
    
    assert len(predictions) == 2

def test_invalid_input(classifier):
    """Test with invalid input."""
    # Test with an empty array
    with pytest.raises(ValueError):
        classifier(images=np.array([]))

# Example of using the asterisk (*) in place of any step keyword:
@pytest.mark.parametrize("image_path", [
    "path_to_image_0.png",
    "path_to_image_1.png",
])
def test_recognize_digits_with_multiple_inputs(classifier, image_path):
    """Test recognizing digits with multiple inputs."""
    *images = [np.array(Image.open(image).convert('L').resize((28, 28))) for image in ["image1", "image2"]]
    
    predictions = classifier(images=np.stack([*images]))
    
    assert len(predictions) == len(*images)
