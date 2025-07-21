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
    """Generates a sample test set of images and their corresponding labels."""
    # For demonstration purposes, we'll use a simple dataset with 10 images.
    # In a real-world scenario, you'd want to replace this with your actual test data.
    num_images = 10
    image_size = (28, 28)
    
    images = np.random.rand(num_images, *image_size).astype(np.uint8)  # Random pixel values for demonstration purposes
    
    labels = np.arange(0, num_images % 10)  # Simple label assignment for demonstration purposes

    return images, labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Tests the accuracy of the digit recognition model on a given test set."""
    
    images, expected_labels = test_set
    
    classifier = ClassifyDigits()
    predicted_labels = np.array(classifier(images))
    
    # Calculate accuracy
    correct_predictions = (predicted_labels == expected_labels).sum()
    accuracy = correct_predictions / len(expected_labels)
    
    assert accuracy > 0.95

class ClassifyDigits(IClassifyDigits):
    def __call__(self, images: np.ndarray) -> np.ndarray:
        """Classifies the given set of digit images using a trained model."""
        
        # Normalize and flatten input data
        normalized_images = images / 255.0
        flattened_images = normalized_images.reshape(-1, 28 * 28)
        
        predictions = model.predict(flattened_images)
        return np.array([int(np.argmax(prediction)) for prediction in predictions])
