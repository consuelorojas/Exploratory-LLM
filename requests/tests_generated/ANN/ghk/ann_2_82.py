import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'ANN'))
from main import ClassifyDigits


# tests/test_digit_recognition.py

import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
from PIL import Image
import tensorflow as tf

@pytest.fixture
def model():
    """Load the trained digit recognition model."""
    return tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Generate a sample test set for testing purposes."""
    # For demonstration, we'll use random images. In real-world scenarios,
    # you would load your actual test dataset here.
    num_images = 1000
    image_size = (28, 28)
    
    images = np.random.rand(num_images, *image_size).astype(np.uint8)
    labels = np.random.randint(10, size=num_images)

    return images, labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test the accuracy of the digit recognition model."""
    # Create an instance of IClassifyDigits
    classifier = ClassifyDigits()

    # Get the test set and its corresponding labels
    images, expected_labels = test_set
    
    # Normalize and flatten the input data for prediction
    normalized_images = (images / 255.0).reshape(-1, 28 * 28)
    
    # Use the model to make predictions on the test set
    predicted_labels = classifier(normalized_images)

    # Calculate accuracy by comparing expected labels with predicted ones
    correct_predictions = np.sum(expected_labels == predicted_labels)
    accuracy = (correct_predictions / len(test_set[0])) * 100

    assert accuracy > 95, f"Expected an accuracy of more than 95%, but got {accuracy:.2f}%"

class ClassifyDigits(IClassifyDigits):
    def __call__(self, images: np.ndarray) -> np.ndarray:
        predictions = model.predict(images.reshape(-1, 28 * 28))
        return np.array([int(np.argmax(prediction)) for prediction in predictions])
