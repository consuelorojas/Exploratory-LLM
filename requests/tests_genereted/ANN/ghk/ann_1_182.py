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
    
    images = np.random.rand(num_samples, *image_size).astype(np.uint8) / 255.0
    labels = np.random.randint(10, size=num_samples)

    return images, labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test the accuracy of the digit recognition model."""
    
    # Create an instance of IClassifyDigits to classify digits.
    class ClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            predictions = model.predict(images.reshape(-1, 28 * 28))
            return np.array([int(np.argmax(prediction)) for prediction in predictions])

    # Get the test set and labels.
    images, labels = test_set

    # Classify the digits using the trained model.
    classifier = ClassifyDigits()
    predicted_labels = classifier(images)

    # Calculate accuracy
    correct_predictions = sum(1 for label, pred_label in zip(labels, predicted_labels) if label == pred_label)
    accuracy = (correct_predictions / len(test_set[0])) * 100

    assert accuracy > 95.0
