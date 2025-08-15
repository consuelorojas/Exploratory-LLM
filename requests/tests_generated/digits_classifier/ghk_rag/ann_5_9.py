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
    # Add more image paths here...
])
def test_recognize_digits(classifier, tmpdir):
    """
    Test that the classifier recognizes digits correctly
    """
    x_test, y_test = load_mnist_test_data()
    
    correct_count = 0
    
    for i in range(10):  # Use at least ten different inputs from dataset
        img_array = np.array(x_test[i])
        
        prediction = classifier(img_array)
        
        if int(prediction) == y_test[i]:
            correct_count += 1
            
    accuracy = (correct_count / len(y_test[:10])) * 100
    
    assert accuracy > 95

def test_classify_digits_invalid_input(classifier):
    """
    Test that the classify function raises an error for invalid input
    """
    with pytest.raises(ValueError):
        classifier(np.array([[[1,2], [3,4]]]))

# Example of using parametrize to generate multiple inputs from dataset
@pytest.mark.parametrize("image_array", [
    np.random.randint(0, 255, size=(28, 28), dtype=np.uint8),
    # Add more image arrays here...
])
def test_classify_digits_valid_input(classifier, image_array):
    """
    Test that the classify function returns a valid output for valid input
    """
    prediction = classifier(image_array)
    
    assert isinstance(prediction, np.ndarray) and len(prediction.shape) == 1

# Example of using fixture to load dataset from file system (if needed)
@pytest.fixture
def image_dataset(tmpdir):
    """Fixture to create an example MNIST-like dataset"""
    for i in range(10):  
        img = Image.new('L', size=(28, 28))
        
        # Save the images as PNG files
        filename = f"image_{i}.png"
        filepath = os.path.join(tmpdir.strpath, filename)
        img.save(filepath)

def test_classify_digits_from_file(classifier, image_dataset):
    """
    Test that the classify function can read and recognize digits from file system
    """
    # Load an example MNIST-like dataset (e.g., 10 images of size 28x28 pixels each) 
    for filename in os.listdir(image_dataset.strpath):  
        img_path = os.path.join(image_dataset.strpath, filename)
        
        with Image.open(img_path).convert('L') as image:
            prediction = classifier(np.array(image))
            
            assert isinstance(prediction, np.ndarray)

