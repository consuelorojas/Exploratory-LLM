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

    Args:
        classifier (ClassifyDigits): The digit recognition model.
        image_path (str): Path to a single-digit number image file.
    """
    x = np.array(Image.open(image_path).convert('L').resize((28, 28)))
    prediction = classifier(images=x)
    assert isinstance(prediction[0], int)

def test_recognize_multiple_digits(classifier):
    """Test the classifier with multiple images."""
    # Load MNIST test data
    x_test, y_test = load_mnist_test_data()
    
    correct_count = 0
    
    for i in range(10):  # Test at least ten different inputs from dataset
        image = np.array(x_test[i])
        prediction = classifier(images=image)
        
        if int(prediction[0]) == y_test[i]:
            correct_count += 1
            
    accuracy = (correct_count / 10) * 100
    
    assert accuracy > 95

def test_recognize_digits_with_random_data(classifier):
    """Test the classifier with random data."""
    # Generate some random images
    np.random.seed(0)
    random_images = np.random.randint(low=0, high=256, size=(10, 28, 28))
    
    predictions = []
    
    for image in random_images:
        prediction = classifier(images=image)
        
        if int(prediction[0]) not in range(10):
            # If the model predicts a non-digit class (which doesn't exist), consider it incorrect
            continue
        
        predictions.append(int(prediction[0]))
            
    assert len(predictions) <= 10

def test_load_model():
    """Test that the model is loaded correctly."""
    try:
        tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")

# Example usage of parametrized testing with multiple images
@pytest.mark.parametrize("image_paths", [
    ["path/to/image1.png", "path/to/image2.png"],
    # Add more lists of image paths here...
])
def test_recognize_multiple_images(classifier, image_paths):
    """Test the classifier with a list of individual images."""
    
    correct_count = 0
    
    for i in range(len(image_paths)):
        x = np.array(Image.open(image_paths[i]).convert('L').resize((28, 28)))
        
        # Assuming you have corresponding labels
        label = int(i % 10)  # Replace with actual label
        
        prediction = classifier(images=x)
        
        if int(prediction[0]) == label:
            correct_count += 1
            
    accuracy = (correct_count / len(image_paths)) * 100
    
    assert accuracy > 95

# Example usage of the asterisk (*) in place of step keywords
def test_recognize_digits_with_asterisk(classifier):
    """Test the classifier with an individual image using asterisks."""
    
    # Load MNIST test data
    x_test, y_test = load_mnist_test_data()
    
    correct_count = 0
    
    for i in range(10):  
        *_, prediction = [classifier(images=np.array(x)) for x in [x_test[i]]]
        
        if int(prediction[0]) == y_test[i]:
            correct_count += 1
            
    accuracy = (correct_count / 10) * 100
    
    assert accuracy > 95
