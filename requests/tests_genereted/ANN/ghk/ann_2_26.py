import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))


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
    """Generate a sample test set of images and their corresponding labels."""
    # For demonstration purposes, we'll use random data.
    np.random.seed(0)
    num_samples = 100
    image_size = (28, 28)
    
    images = np.random.randint(0, 256, size=(num_samples, *image_size), dtype=np.uint8)
    labels = np.random.randint(0, 10, size=num_samples)

    return images, labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test the accuracy of the digit recognition model."""
    
    # Create an instance of IClassifyDigits
    classifier = ClassifyDigits()
    
    # Get the test set and its corresponding labels
    images, expected_labels = test_set
    
    # Convert images to grayscale (L) mode as required by the classify method
    gray_images = np.array([np.array(Image.fromarray(image).convert('L')) for image in images])
    
    # Classify the test set using the model
    predicted_labels = classifier(gray_images)
    
    # Calculate accuracy
    correct_predictions = sum(1 for pred, exp in zip(predicted_labels, expected_labels) if pred == exp)
    accuracy = (correct_predictions / len(expected_labels)) * 100
    
    assert accuracy > 95

class ClassifyDigits(IClassifyDigits):
    def __call__(self, images: np.ndarray) -> np.ndarray:
        # Load the model
        loaded_model = tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)
        
        # Normalize and flatten images as required by the classify method
        normalized_images = images / 255.0
        flattened_images = normalized_images.reshape(-1, 28 * 28)

        predictions = loaded_model.predict(flattened_images)
        return np.array([int(np.argmax(prediction)) for prediction in predictions])
