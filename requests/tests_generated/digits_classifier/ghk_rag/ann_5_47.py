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

    :param classifier: A ClassifyDigits instance.
    :param image_path: The path to a single-digit number image file (0-9).
    """
    # Load and preprocess the image
    img = Image.open(image_path).convert('L').resize((28, 28))
    images = np.array(img)

    # Make predictions using the classifier
    prediction = classifier(images=images)
    
    # Assuming we have a way to get the actual digit from the file name or another source...
    expected_digit = int(os.path.basename(image_path).split('.')[0])  # Replace with your logic

    assert prediction == [expected_digit]

def test_recognize_multiple_digits(classifier):
    """
    Test the classifier's accuracy on multiple images.

    :param classifier: A ClassifyDigits instance.
    """
    x_test, y_test = load_mnist_test_data()
    
    correct_count = 0
    total_images = len(x_test)

    for i in range(total_images):
        img = np.array([x_test[i]])
        
        # Make predictions using the classifier
        prediction = classifier(images=img)
        
        if prediction[0] == y_test[i]:
            correct_count += 1

    accuracy = (correct_count / total_images) * 100
    
    assert accuracy > 95, f"Expected an accuracy of at least 95%, but got {accuracy:.2f}%"

def test_recognize_random_strings(classifier):
    """
    Test the classifier's behavior on non-digit images.

    :param classifier: A ClassifyDigits instance.
    """
    # Create a random image that is not a digit
    img = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)
    
    prediction = classifier(images=img)

    assert isinstance(prediction[0], int) and 0 <= prediction[0] < 10

def test_load_model():
    """
    Test that the model is loaded correctly.
    """
    try:
        tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")
