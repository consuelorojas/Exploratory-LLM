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
    # For demonstration, we'll use 10 random images with known labels.
    np.random.seed(0)
    images = np.random.rand(10, 28, 28) * 255.0
    labels = np.random.randint(0, 9, size=10)

    return images.astype(np.uint8), labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test the accuracy of digit recognition using a trained model."""
    # Create an instance of IClassifyDigits
    classifier = ClassifyDigits()

    # Get the test set and its corresponding labels.
    images, expected_labels = test_set

    # Convert images to grayscale and resize them if necessary.
    resized_images = np.array([Image.fromarray(image).convert('L').resize((28, 28)) for image in images])

    # Classify the digits using the trained model
    predicted_labels = classifier(resized_images)

    # Calculate accuracy by comparing expected labels with predicted ones.
    correct_predictions = sum(1 for pred_label, exp_label in zip(predicted_labels, expected_labels) if pred_label == exp_label)
    accuracy = (correct_predictions / len(expected_labels)) * 100

    assert accuracy > 95
