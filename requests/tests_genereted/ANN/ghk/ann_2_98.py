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
import os


@pytest.fixture
def model():
    """Loads the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generates a sample test set for testing purposes."""
    # For demonstration, we'll use 10 random images with known labels.
    num_images = 10
    image_size = (28, 28)
    images = np.random.rand(num_images, *image_size).astype(np.uint8)  # Random pixel values between 0 and 255

    # Assigning dummy labels for demonstration purposes. In a real scenario,
    # these would be the actual expected classifications.
    labels = np.arange(10)

    return images, labels


def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Tests if the digit recognition model achieves an accuracy of more than 95% on the given test set."""
    
    class ClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            # Normalize and flatten input data
            normalized_images = images / 255.0
            flattened_images = normalized_images.reshape(-1, 28 * 28)

            predictions = model.predict(flattened_images)
            return np.array([int(np.argmax(prediction)) for prediction in predictions])

    classifier: IClassifyDigits = ClassifyDigits()
    
    # Load test images and labels from the fixture
    test_images, expected_labels = test_set
    
    predicted_labels = classifier(test_images)

    accuracy = sum(1 for pred_label, exp_label in zip(predicted_labels, expected_labels) if pred_label == exp_label) / len(expected_labels)
    
    assert accuracy > 0.95

