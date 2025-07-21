import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'ANN'))
from main import ClassifyDigits


# tests/test_digit_recognition.py

import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
from PIL.Image import Image
import tensorflow as tf
from sklearn.metrics import accuracy_score


@pytest.fixture
def model():
    """Loads the trained digit recognition model."""
    return tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generates a sample test set for testing purposes."""
    # Replace this with your actual test data loading logic.
    images = np.random.rand(100, 28, 28)
    labels = np.random.randint(0, 10, size=100)
    return images, labels


@pytest.fixture
def classifier(model):
    """Creates an instance of the digit classification class."""
    class ClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            images = images / 255.0                 # normalize
            images = images.reshape(-1, 28 * 28)    # flatten

            predictions = model.predict(images)
            return np.array([int(np.argmax(prediction)) for prediction in predictions])

    return ClassifyDigits()


def test_digit_recognition_accuracy(classifier: IClassifyDigits, test_set):
    """Tests the accuracy of the digit recognition model."""
    images, labels = test_set
    predicted_labels = classifier(images)

    # Calculate accuracy using sklearn's accuracy_score function.
    accuracy = accuracy_score(labels, predicted_labels)
    
    assert accuracy > 0.95, f"Expected accuracy to be more than 95%, but got {accuracy:.2f}"
