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
    
    # Extract the images and their corresponding labels from the test set.
    images, expected_labels = test_set
    
    # Create an instance of IClassifyDigits to classify the digits using our model.
    classifier = ClassifyDigits()
    
    # Normalize and flatten the input data for classification
    normalized_images = (images / 255.0).reshape(-1, 28 * 28)
    
    # Use the loaded model directly instead of relying on IClassifyDigits' internal implementation details.
    predictions = np.array([int(np.argmax(model.predict(normalized_image.reshape(1, -1)))) for normalized_image in normalized_images])
    
    # Calculate accuracy
    correct_predictions = sum(predictions == expected_labels)
    accuracy = (correct_predictions / len(expected_labels)) * 100
    
    assert accuracy > 95

class ClassifyDigits:
    def __call__(self, images: np.ndarray) -> np.ndarray[np.int_]:
        model = load_model(MODEL_DIGIT_RECOGNITION_PATH)

        # Normalize and flatten the input data for classification
        normalized_images = (images / 255.0).reshape(-1, 28 * 28)
        
        predictions = model.predict(normalized_images)
        return np.array([int(np.argmax(prediction)) for prediction in predictions])
