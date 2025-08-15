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
    # Add more image paths as needed
])
def test_recognize_digits(classifier, image_path):
    """
    Test the classifier with a single input from the dataset.
    
    :param classifier: A ClassifyDigits instance
    :param image_path: Path to an image file containing a digit
    """
    # Load and preprocess the image data
    img = Image.open(image_path).convert('L').resize((28, 28))
    images = np.array(img)
    
    # Get the predicted output from the classifier
    prediction = classifier(images=images)[0]
    
    # Assuming we have a way to get the actual digit value for comparison (e.g., filename or label)
    expected_digit = int(os.path.basename(image_path).split('.')[0])  # Replace with your logic
    
    assert prediction == expected_digit

def test_recognize_digits_accuracy(classifier):
    """
    Test that the classifier recognizes over 95% of digits correctly.
    
    :param classifier: A ClassifyDigits instance
    """
    x_test, y_test = load_mnist_test_data()
    num_correct = 0
    
    for i in range(10):  # Use at least ten different inputs from your dataset
        img = x_test[i]
        images = np.array(img) / 255.0
        prediction = classifier(images=images)[0]
        
        if prediction == y_test[i]:
            num_correct += 1
    
    accuracy = (num_correct / len(y_test[:10])) * 100
    assert accuracy > 95

def test_classify_digits_interface(classifier):
    """
    Test that the ClassifyDigits instance conforms to the IClassifyDigits interface.
    
    :param classifier: A ClassifyDigits instance
    """
    # Assuming we have a way to create an NDArray of images (e.g., using numpy)
    images = np.random.rand(1, 28 * 28)  # Replace with your logic
    
    result = classifier(images=images)
    
    assert isinstance(result, np.ndarray)

def test_classify_digits_model_loaded():
    """
    Test that the model is loaded correctly.
    
    :param model_path: Path to the saved Keras model
    """
    try:
        tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        pytest.fail(f"Failed to load model from {model_path}: {str(e)}")
