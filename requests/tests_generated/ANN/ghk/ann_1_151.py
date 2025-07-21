import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'ANN'))
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
    
    # Generate some dummy data. In real-world scenarios, you'd load actual test data here.
    images = np.random.rand(num_images, *image_size).astype(np.uint8) 
    labels = np.arange(0, num_images)

    return images, labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Tests the accuracy of digit recognition model."""
    
    # Initialize our classifier
    class ClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            images = images / 255.0                 # normalize
            images = images.reshape(-1, 28 * 28)    # flatten

            predictions = model.predict(images)
            return np.array([int(np.argmax(prediction)) for prediction in predictions])

    classifier = ClassifyDigits()

    test_images, expected_labels = test_set
    
    predicted_labels = classifier(test_images)

    accuracy = sum(1 for pred_label, exp_label in zip(predicted_labels, expected_labels) if pred_label == exp_label) / len(expected_labels)
    
    assert accuracy > 0.95
