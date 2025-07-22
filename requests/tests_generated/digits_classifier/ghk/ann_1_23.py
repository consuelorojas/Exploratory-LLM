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
    # Initialize the classifier
    classifier = ClassifyDigits()

    # Get the test set
    images, expected_labels = test_set
    
    # Normalize and flatten the images for classification
    normalized_images = (images / 255.0).reshape(-1, 28 * 28)

    # Use the model to classify the test set directly since we have it loaded already
    predictions = np.array([int(np.argmax(model.predict(normalized_image.reshape(1, -1)))) 
                            for normalized_image in normalized_images])

    # Calculate accuracy
    correct_predictions = sum(predictions == expected_labels)
    accuracy = (correct_predictions / len(expected_labels)) * 100

    assert accuracy > 95


class ClassifyDigits:
    def __call__(self, images: np.ndarray) -> np.ndarray[np.int_]:
        """Classifies the given images using a loaded model."""
        
        # Load the trained digit recognition model
        model = tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)

        normalized_images = (images / 255.0).reshape(-1, 28 * 28)
        predictions = np.array([int(np.argmax(model.predict(normalized_image.reshape(1, -1)))) 
                                for normalized_image in normalized_images])
        
        return predictions
