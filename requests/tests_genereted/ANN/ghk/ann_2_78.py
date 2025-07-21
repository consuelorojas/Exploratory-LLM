import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))


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
    
    # Generate some dummy data for testing. In real-world scenarios,
    # you would load your actual test dataset here.
    images = np.random.rand(num_images, *image_size).astype(np.uint8)
    labels = np.random.randint(0, 10, num_images)

    return images, labels


def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Tests the accuracy of the digit recognition model."""
    
    # Get the test set
    images, expected_labels = test_set
    
    # Create an instance of IClassifyDigits to classify digits.
    classifier = ClassifyDigits()
    
    # Convert PIL Image objects into numpy arrays for classification.
    image_arrays = np.array([np.array(image) for image in [Image.fromarray(img) for img in images]])
    
    # Get the predicted labels
    predicted_labels = classifier(images=image_arrays)
    
    # Calculate accuracy by comparing expected and actual labels.
    correct_predictions = sum(1 for pred, exp in zip(predicted_labels, expected_labels) if pred == exp)
    accuracy = (correct_predictions / len(expected_labels)) * 100
    
    assert accuracy > 95
