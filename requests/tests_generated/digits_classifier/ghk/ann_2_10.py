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
    """Load the trained digit recognition model."""
    return tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Generate a sample test set for testing purposes."""
    # For demonstration, we'll use random images. In real-world scenarios,
    # you would load your actual test dataset here.
    num_images = 1000
    image_size = (28, 28)
    
    images = np.random.rand(num_images, *image_size).astype(np.uint8)
    labels = np.random.randint(10, size=num_images)

    return images, labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test the accuracy of the digit recognition model."""
    # Create an instance of IClassifyDigits
    classifier = ClassifyDigits()
    
    # Get the test set
    images, expected_labels = test_set
    
    # Make predictions using the model
    predicted_labels = np.array([int(np.argmax(prediction)) for prediction in 
                                 model.predict(images / 255.0).reshape(-1, 10)])
    
    # Calculate accuracy
    correct_predictions = sum(predicted_labels == expected_labels)
    accuracy = (correct_predictions / len(expected_labels)) * 100
    
    assert accuracy > 95

class ClassifyDigits(IClassifyDigits):
    def __call__(self, images: np.ndarray) -> np.ndarray:
        """Classify the given images using the model."""
        # Normalize and flatten the input
        normalized_images = images / 255.0
        
        predictions = tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH).predict(normalized_images.reshape(-1, 28 * 28))
        
        return np.array([int(np.argmax(prediction)) for prediction in predictions])
