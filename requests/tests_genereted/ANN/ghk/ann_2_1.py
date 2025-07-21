import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))


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
    """Generates a sample test set of images with known labels."""
    # For this example, we'll use 10 random images from MNIST dataset.
    import tensorflow as tf
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    return x_train[:10], y_train[:10]

def test_digit_recognition_accuracy(model: load_model, test_set):
    """Tests the accuracy of digit recognition model."""
    
    # Create an instance of ClassifyDigits
    class ClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            images = images / 255.0                 # normalize
            images = images.reshape(-1, 28 * 28)    # flatten

            predictions = model.predict(images)
            return np.array([int(np.argmax(prediction)) for prediction in predictions])

    classify_digits = ClassifyDigits()

    test_images, expected_labels = test_set
    
    predicted_labels = classify_digits(test_images)

    accuracy = sum(1 for pred, exp in zip(predicted_labels, expected_labels) if pred == exp) / len(expected_labels)
    
    assert accuracy > 0.95
