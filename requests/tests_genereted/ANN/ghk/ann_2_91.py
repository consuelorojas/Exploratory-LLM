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
    # For demonstration purposes, we'll create two simple 28x28 grayscale images.
    image1 = np.zeros((28, 28))
    image2 = np.ones((28, 28))

    label1 = 0
    label2 = 1

    return [(image1, label1), (image2, label2)]


@pytest.fixture
def classifier():
    """Creates an instance of the digit classification class."""
    from digits_classifier import ClassifyDigits
    return ClassifyDigits()


def test_digit_recognition_accuracy(model: load_model, test_set, classifier):
    """
    Tests that the trained model achieves more than 95% accuracy on a given test set.

    Args:
        model (load_model): The loaded digit recognition model.
        test_set (list[tuple[np.ndarray, int]]): A list of tuples containing images and their labels.
        classifier (IClassifyDigits): An instance of the ClassifyDigits class.
    """
    # Extract images from the test set
    images = np.array([image for image, _ in test_set])

    # Normalize and flatten the images
    normalized_images = images / 255.0
    flattened_images = normalized_images.reshape(-1, 28 * 28)

    # Make predictions using the classifier instance (not directly with model.predict)
    predicted_labels = np.array([int(np.argmax(prediction)) for prediction in 
                                 [classifier(images=np.expand_dims(image, axis=0))[0] for image in images]])

    actual_labels = np.array([label for _, label in test_set])

    # Calculate accuracy
    correct_predictions = sum(predicted_labels == actual_labels)
    total_images = len(test_set)

    assert (correct_predictions / total_images) > 0.95, "Model did not achieve more than 95% accuracy"
