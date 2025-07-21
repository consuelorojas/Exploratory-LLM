import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))


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
    """Generate a sample test set of images and their corresponding labels."""
    # For demonstration purposes, we'll use random data.
    np.random.seed(0)
    num_samples = 100
    image_size = (28, 28)
    
    images = np.random.randint(low=0, high=256, size=(num_samples, *image_size), dtype=np.uint8)
    labels = np.random.randint(low=0, high=10, size=num_samples)

    return images, labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test the accuracy of the digit recognition model."""
    
    # Load and normalize the test set
    images, expected_labels = test_set
    
    classifier = IClassifyDigits()
    predicted_labels = np.array([int(np.argmax(prediction)) for prediction in 
                                 classifier(images / 255.0).reshape(-1)])
    
    accuracy = sum(predicted_labels == expected_labels) / len(expected_labels)
    
    assert accuracy > 0.95, f"Expected an accuracy of more than 95%, but got {accuracy:.2f}"
