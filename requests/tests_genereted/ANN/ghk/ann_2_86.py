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
    """Load the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generate a sample test set of images and their corresponding labels."""
    # For demonstration purposes, let's assume we have 10 images in our test set.
    num_images = 10

    # Generate random image data (28x28 grayscale).
    images = np.random.rand(num_images, 28, 28)

    # Assign some arbitrary labels to these images for testing purposes.
    labels = np.array([i % 10 for i in range(num_images)])

    return images, labels


def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test the accuracy of digit recognition using a trained model."""
    images, expected_labels = test_set

    # Create an instance of IClassifyDigits to classify digits.
    classifier = ClassifyDigits()

    # Normalize and flatten image data before classification.
    normalized_images = images / 255.0
    flattened_images = normalized_images.reshape(-1, 28 * 28)

    # Use the model directly for prediction since we have it available in this context.
    predictions = np.array([int(np.argmax(model.predict(image.reshape(1, -1)))) for image in flattened_images])

    # Calculate accuracy by comparing predicted labels with expected labels.
    correct_predictions = sum(predictions == expected_labels)
    accuracy = (correct_predictions / len(expected_labels)) * 100

    assert accuracy > 95
