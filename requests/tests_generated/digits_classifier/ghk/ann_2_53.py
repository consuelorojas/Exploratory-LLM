import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






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
    # For demonstration purposes, we'll use 10 random images.
    num_images = 10
    image_size = (28, 28)
    
    # Generate some dummy data for testing. In real-world scenarios,
    # you would load your actual test dataset here.
    images = np.random.rand(num_images, *image_size).astype(np.uint8)
    labels = np.random.randint(0, 10, num_images)

    return images, labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Tests the accuracy of the digit recognition model."""
    
    # Load the classifier
    class Classifier(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            images = images / 255.0                 # normalize
            images = images.reshape(-1, 28 * 28)    # flatten

            predictions = model.predict(images)
            return np.array([int(np.argmax(prediction)) for prediction in predictions])

    classifier = Classifier()

    # Get the test set and labels
    images, expected_labels = test_set
    
    # Classify the test set using the loaded model
    predicted_labels = classifier(images)

    # Calculate accuracy
    correct_predictions = sum(1 for pred, exp in zip(predicted_labels, expected_labels) if pred == exp)
    accuracy = (correct_predictions / len(expected_labels)) * 100

    assert accuracy > 95.0, f"Expected an accuracy of more than 95%, but got {accuracy:.2f}%"
