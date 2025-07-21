import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))
from main import ClassifyDigits


# tests/test_digit_recognition.py

import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
from tensorflow.keras.models import load_model
from PIL.Image import open, Image
import os


@pytest.fixture
def model():
    """Loads the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generates a sample test set of images with known labels."""
    # For demonstration purposes, we'll use 10 random images.
    num_images = 10
    image_size = (28, 28)
    
    # Generate some dummy data for testing. Replace this with actual test data in your project structure.
    images = np.random.rand(num_images, *image_size).astype(np.uint8) 
    labels = np.random.randint(0, 9, num_images)

    return images, labels


def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Tests the accuracy of the digit recognition model."""
    
    # Load and normalize the test set
    images, expected_labels = test_set
    
    classifier = IClassifyDigits()
    predicted_labels = np.array([int(np.argmax(classifier(images / 255.0).reshape(-1))) for _ in range(len(expected_labels))])
    
    accuracy = sum(predicted_labels == expected_labels) / len(expected_labels)
    
    assert accuracy > 0.95, f"Expected an accuracy of more than 95%, but got {accuracy:.2f}"
