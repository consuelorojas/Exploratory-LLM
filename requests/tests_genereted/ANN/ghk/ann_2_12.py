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
from PIL import Image


@pytest.fixture
def model():
    """Load the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generate a sample test set of images and their corresponding labels."""
    # For demonstration purposes, we'll use random data.
    np.random.seed(0)
    num_samples = 100
    image_size = (28, 28)

    images = np.random.randint(0, 256, size=(num_samples, *image_size), dtype=np.uint8)
    labels = np.random.randint(0, 10, size=num_samples)

    return images, labels


@pytest.fixture
def classifier():
    """Create an instance of the digit classification class."""
    from digits_classifier import ClassifyDigits

    return ClassifyDigits()


def test_digit_recognition_accuracy(model: tf.keras.Model, test_set, classifier):
    """
    Test that the trained model achieves more than 95% accuracy on a sample test set.

    Given:
        - A trained digit recognition model
        - A test set of images and their corresponding labels

    When:
        - The test set is classified using the trained model

    Then:
        - An accuracy of more than 95 percent should be achieved.
    """
    # Load the sample test data (images, labels)
    images, expected_labels = test_set
    num_samples = len(images)

    # Preprocess and classify each image in the test set
    predicted_labels = classifier(np.array([Image.fromarray(image).convert('L').resize((28, 28)) for image in images]))

    # Calculate accuracy by comparing predicted labels with actual labels
    correct_predictions = np.sum(predicted_labels == expected_labels)
    accuracy = (correct_predictions / num_samples) * 100

    assert accuracy > 95.0


def test_digit_recognition_model_output(model: tf.keras.Model, classifier):
    """
    Test that the trained model produces a valid output for an example input.

    Given:
        - A trained digit recognition model
        - An instance of the ClassifyDigits class

    When:
        - The classify method is called with sample data

    Then:
        - The predicted label should be within the range [0, 9].
    """
    # Generate a random example image (28x28)
    np.random.seed(42)  # For reproducibility
    example_image = np.random.randint(0, 256, size=(1, 28 * 28), dtype=np.uint8)

    predicted_label = classifier(example_image)[0]

    assert isinstance(predicted_label, int) and (predicted_label >= 0 and predicted_label <= 9)
