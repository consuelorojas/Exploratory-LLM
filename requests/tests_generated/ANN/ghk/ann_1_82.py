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
    """Fixture to load the trained digit recognition model."""
    return tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Fixture to generate a sample test set for testing purposes.
    
    Note: In real-world scenarios, this should be replaced with an actual test dataset.
    """
    # Generate random images and labels (replace with your own data)
    np.random.seed(0)  # For reproducibility
    num_samples = 100
    image_size = (28, 28)
    
    images = np.random.randint(low=0, high=256, size=(num_samples, *image_size), dtype=np.uint8)
    labels = np.random.randint(low=0, high=10, size=num_samples)

    return images, labels


def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test the accuracy of digit recognition using a trained model."""
    
    # Load and prepare data
    images, expected_labels = test_set
    
    classifier = IClassifyDigits()
    
    predicted_labels = np.array([int(np.argmax(prediction)) for prediction in 
                                 classifier(images / 255.0).reshape(-1)])
    
    accuracy = accuracy_score(expected_labels, predicted_labels)
    
    assert accuracy > 0.95
