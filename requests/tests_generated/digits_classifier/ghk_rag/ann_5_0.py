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
    assert isinstance(prediction[0], int) and 0 <= prediction[0] <= 9

def test_recognize_digits_accuracy(classifier):
    """
    Test the accuracy of the digit recognition model
    """
    x_test, y_test = load_mnist_test_data()
    
    # Preprocess images for testing
    num_images_to_use = min(10000, len(x_test))  # Use up to 10k test images
    selected_indices = np.random.choice(len(x_test), size=num_images_to_use)
    x_selected = x_test[selected_indices]
    y_selected = y_test[selected_indices]

    predictions = []
    for image in x_selected:
        img_array = np.array(image).reshape(1, 28 * 28) / 255.0
        prediction = classifier(images=img_array)[0]
        predictions.append(prediction)

    accuracy = sum([p == t for p, t in zip(predictions, y_selected)]) / len(y_selected)
    
    assert accuracy > 0.95

def test_classify_digits_interface(classifier):
    """
    Test the ClassifyDigits interface
    """
    images = np.random.rand(1, 28 * 28)  # Create a random image array
    
    prediction = classifier(images=images)
    assert isinstance(prediction[0], int)

# Example of using parametrize to test multiple inputs at once:
@pytest.mark.parametrize("input_image", [
    (np.array(Image.new('L', (28, 28), 'white'))),
    # Add more input images here...
])
def test_classify_digits_multiple_inputs(classifier, input_image):
    """
    Test the classifier with an individual image
    """
    prediction = classifier(images=input_image)
    assert isinstance(prediction[0], int) and 0 <= prediction[0] <= 9

# Example of using parametrize to test multiple inputs at once:
@pytest.mark.parametrize("input_images", [
    (np.random.rand(1, 28 * 28)),
    # Add more input images here...
])
def test_classify_digits_multiple_inputs_array(classifier, input_images):
    """
    Test the classifier with an array of individual image
    """
    prediction = classifier(images=input_images)
    assert isinstance(prediction[0], int) and 0 <= prediction[0] <= 9

# Example usage:
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--durations=10"])
