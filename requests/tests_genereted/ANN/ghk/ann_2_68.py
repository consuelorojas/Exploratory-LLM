import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))
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
    # For demonstration, we'll use random images. In real-world scenarios,
    # you'd load your actual test dataset here.
    num_samples = 1000
    image_size = (28, 28)
    
    images = np.random.rand(num_samples, *image_size).astype(np.uint8)
    labels = np.random.randint(10, size=num_samples)

    return images, labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test the accuracy of digit recognition model."""
    # Arrange
    classifier = ClassifyDigits()
    
    # Act
    predictions = classifier(test_set[0])
    accuracy = np.mean(predictions == test_set[1])

    # Assert
    assert accuracy > 0.95

class ClassifyDigits(IClassifyDigits):
    def __call__(self, images: np.ndarray) -> np.ndarray:
        """Classify the given images using the trained model."""
        
        images = images / 255.0                 # normalize
        images = images.reshape(-1, 28 * 28)    # flatten

        predictions = model.predict(images)
        return np.array([int(np.argmax(prediction)) for prediction in predictions])
