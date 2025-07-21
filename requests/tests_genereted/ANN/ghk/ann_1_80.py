import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))
from main import ClassifyDigits


# tests/test_digit_recognition.py

import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
import tensorflow as tf
from PIL import Image


@pytest.fixture
def model():
    """Load the trained digit recognition model."""
    return tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generate a sample test set for testing purposes."""
    # For demonstration, we'll use random images. In real-world scenarios,
    # you'd load your actual test dataset.
    num_samples = 1000
    image_size = (28, 28)
    
    # Generate dummy data with labels
    x_test = np.random.rand(num_samples, *image_size).astype(np.float32) / 255.0
    y_test = np.random.randint(10, size=num_samples)

    return x_test.reshape(-1, 28*28), y_test


def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test the accuracy of digit recognition model."""
    
    # Initialize classifier with loaded model
    class ClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            predictions = model.predict(images.reshape(-1, 28*28))
            return np.array([int(np.argmax(prediction)) for prediction in predictions])

    x_test, y_test = test_set
    classifier = ClassifyDigits()
    
    # Make predictions on the test set using our loaded model and custom classifier
    predicted_labels = classifier(x_test)
    
    # Calculate accuracy by comparing true labels with predicted ones
    correct_predictions = np.sum(predicted_labels == y_test)
    accuracy = (correct_predictions / len(y_test)) * 100
    
    assert accuracy > 95, f"Expected an accuracy of more than 95%, but got {accuracy:.2f}%"
