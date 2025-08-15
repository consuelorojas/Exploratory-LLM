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
    "path/to/image1.png",
    "path/to/image2.png",
    # Add more image paths as needed for testing
])
def test_recognize_digits(classifier, image_path):
    """Test recognizing digits with a single input"""
    x = PIL.Image.open(image_path).convert('L').resize((28, 28))
    images = np.array(x)
    prediction = classifier(images=images)
    assert isinstance(prediction, int)

def test_accuracy_with_mnist_data(classifier):
    """Test accuracy of digit recognition using MNIST dataset"""
    # Load MNIST data
    x_test, y_test = load_mnist_test_data()
    
    correct_predictions = 0
    
    for i in range(10):  # Test with at least ten different inputs from the dataset
        image = x_test[i]
        images = np.array(image)
        
        prediction = classifier(images=images)[0]  # Get the predicted digit
        
        if prediction == y_test[i]:
            correct_predictions += 1
    
    accuracy = (correct_predictions / len(y_test[:10])) * 100
    assert accuracy > 95

def test_recognize_multiple_digits(classifier):
    """Test recognizing multiple digits"""
    x, _ = load_mnist_test_data()
    
    images_list = []
    for i in range(10):  
        image = PIL.Image.fromarray(x[i]).convert('L').resize((28, 28))
        images_list.append(np.array(image))

    # Stack the list of arrays into a single array
    stacked_images = np.stack(images_list)
    
    predictions = classifier(stacked_images)
    
    assert len(predictions) == 10

def test_invalid_input(classifier):
    """Test handling invalid input"""
    with pytest.raises(Exception):
        classifier(np.array([1,2])) # Invalid shape or size of the image
