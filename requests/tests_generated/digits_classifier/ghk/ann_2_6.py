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
    
    # Generate dummy labels and images
    labels = np.random.randint(10, size=num_images)  # Random digit classes for demonstration
    images = np.random.rand(num_images, *image_size).astype(np.uint8)

    return images, labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test the accuracy of the digit recognition model."""
    
    class ClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            # Normalize and flatten images
            normalized_images = images / 255.0
            flattened_images = normalized_images.reshape(-1, 28 * 28)
            
            predictions = model.predict(flattened_images)
            return np.array([int(np.argmax(prediction)) for prediction in predictions])

    classifier: IClassifyDigits = ClassifyDigits()
    
    # Load test images and labels from fixture
    test_images, expected_labels = test_set
    
    predicted_classes = classifier(test_images)

    accuracy = sum(predicted_classes == expected_labels) / len(expected_labels)
    
    assert accuracy > 0.95, f"Expected model to have an accuracy of more than 95%, but got {accuracy:.2f}"
