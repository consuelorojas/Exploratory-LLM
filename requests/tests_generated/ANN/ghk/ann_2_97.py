import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'ANN'))
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
    num_images = 1000
    image_size = (28, 28)
    
    images = np.random.rand(num_images, *image_size).astype(np.uint8)
    labels = np.random.randint(10, size=num_images)

    return images, labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test the accuracy of digit recognition using a trained model."""
    
    # Create an instance of IClassifyDigits
    classifier = ClassifyDigits()
    
    # Get the test set and its corresponding labels
    images, expected_labels = test_set
    
    # Convert images to grayscale (L) mode as required by the model
    gray_images = np.array([np.array(Image.fromarray(image).convert('L')) for image in images])
    
    # Classify the digits using the trained model
    predicted_labels = classifier(gray_images)
    
    # Calculate accuracy
    correct_predictions = sum(1 for pred, exp in zip(predicted_labels, expected_labels) if pred == exp)
    accuracy = (correct_predictions / len(expected_labels)) * 100
    
    assert accuracy > 95

class ClassifyDigits(IClassifyDigits):
    def __call__(self, images: np.ndarray) -> np.ndarray:
        # Normalize and flatten the input
        normalized_images = images.astype(np.float32) / 255.0
        
        predictions = model.predict(normalized_images.reshape(-1, 28 * 28))
        
        return np.array([int(np.argmax(prediction)) for prediction in predictions])
