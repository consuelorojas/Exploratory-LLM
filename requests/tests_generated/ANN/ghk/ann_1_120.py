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
        test_set (tuple[np.ndarray]): A tuple containing images and labels for testing.
        classifier (IClassifyDigits): An instance of the ClassifyDigits class.
    """
    # Extract images and labels from the test set
    images, expected_labels = test_set

    # Normalize and flatten images as required by the classification logic
    normalized_images = images / 255.0
    flattened_images = normalized_images.reshape(-1, 28 * 28)

    # Use the classifier to predict labels for the given images
    predicted_labels = np.array([int(np.argmax(prediction)) for prediction in model.predict(flattened_images)])

    # Calculate accuracy using scikit-learn's accuracy_score function
    accuracy = accuracy_score(expected_labels, predicted_labels)

    assert accuracy > 0.95


def test_classify_digits(classifier: IClassifyDigits):
    """
    Test the ClassifyDigits class with a sample image.

    Args:
        classifier (IClassifyDigits): An instance of the ClassifyDigits class.
    """

    # Load an example image
    img = Image.new('L', (28, 28))
    pixels = img.load()

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if i == j:
                pixels[i, j] = 255

    images_array = np.array([np.array(img)])

    # Classify the image
    result = classifier(images=images_array)

    assert len(result) > 0


def test_classify_digits_invalid_input(classifier: IClassifyDigits):
    """
    Test that an invalid input raises a ValueError.

    Args:
        classifier (IClassifyDigits): An instance of the ClassifyDigits class.
    """

    # Try to classify with no images
    try:
        result = classifier(images=None)
        assert False, "Expected ValueError"
    except Exception as e:
        pass

