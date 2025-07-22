import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






# tests/test_digit_recognition.py

import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
from PIL import Image
import tensorflow as tf

@pytest.fixture
def model():
    """Fixture to load the trained digit recognition model."""
    return tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Fixture to generate a sample test set of images."""
    # Generate 10 random images with labels for testing purposes.
    np.random.seed(0)
    images = np.random.rand(10, 28, 28) * 255.0
    labels = np.random.randint(0, 9, size=10)

    return images.astype(np.uint8), labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test the accuracy of the digit recognition model."""
    # Load and initialize the classifier.
    class ClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            images = images / 255.0
            images = images.reshape(-1, 28 * 28)
            predictions = model.predict(images)
            return np.array([int(np.argmax(prediction)) for prediction in predictions])

    # Get the test set.
    images, labels = test_set

    # Classify the test set using the loaded model.
    classifier = ClassifyDigits()
    predicted_labels = classifier(images)

    # Calculate accuracy.
    correct_predictions = sum(1 for label, pred_label in zip(labels, predicted_labels) if label == pred_label)
    accuracy = (correct_predictions / len(test_set[0])) * 100

    assert accuracy > 95
