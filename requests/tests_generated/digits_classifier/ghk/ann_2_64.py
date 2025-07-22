import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






# tests/test_digit_recognition.py

import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import accuracy_score


@pytest.fixture
def model():
    """Load the trained digit recognition model."""
    return tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generate a sample test set for demonstration purposes."""
    # Replace this with your actual test data loading logic.
    images = np.random.rand(100, 28, 28)
    labels = np.random.randint(0, 10, size=100)
    return images, labels


@pytest.fixture
def classifier():
    """Create an instance of the digit classification class."""
    from digits_classifier import ClassifyDigits
    return ClassifyDigits()


def test_digit_recognition_accuracy(model: tf.keras.Model, test_set, classifier: IClassifyDigits):
    """
    Test that the trained model achieves more than 95% accuracy on a given test set.

    Args:
        model (tf.keras.Model): The loaded digit recognition model.
        test_set (tuple[np.ndarray]): A tuple containing images and labels of the test data.
        classifier (IClassifyDigits): An instance of the ClassifyDigits class for making predictions.
    """
    # Extract images and labels from the test set
    images, labels = test_set

    # Normalize and flatten the input images as required by the model
    normalized_images = images / 255.0
    flattened_images = normalized_images.reshape(-1, 28 * 28)

    # Make predictions using the classifier instance (which uses the loaded model internally)
    predicted_labels = np.array([int(np.argmax(prediction)) for prediction in classifier(flattened_images)])

    # Calculate accuracy of the model on this test set
    accuracy = accuracy_score(labels, predicted_labels)

    assert accuracy > 0.95, f"Expected an accuracy above 95%, but got {accuracy:.2f}"
