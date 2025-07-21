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
from sklearn.metrics import accuracy_score


@pytest.fixture
def model():
    """Load the trained digit recognition model."""
    return tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generate a sample test set for demonstration purposes.
    
    In a real-world scenario, this would be replaced with an actual test dataset.
    """
    # Generate some random images and labels (this is just a placeholder)
    np.random.seed(0)  # For reproducibility
    num_samples = 100
    image_size = (28, 28)
    
    images = np.random.randint(low=0, high=256, size=(num_samples,) + image_size).astype(np.uint8)
    labels = np.random.randint(low=0, high=10, size=num_samples)

    return images, labels


def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test the accuracy of the digit recognition model."""
    
    # Extract the images and labels from the test set
    images, expected_labels = test_set
    
    # Create an instance of IClassifyDigits to classify the digits
    classifier = ClassifyDigits()
    
    # Normalize and flatten the input data (as per the implementation in ClassifyDigits)
    normalized_images = images / 255.0
    flattened_images = normalized_images.reshape(-1, 28 * 28)

    # Use the model directly for prediction to avoid potential issues with image loading
    predictions = np.array([int(np.argmax(model.predict(image.reshape(1, -1)))) for image in flattened_images])
    
    # Calculate the accuracy of the classifier using sklearn's accuracy_score function
    actual_labels = expected_labels  # For clarity and readability
    
    accuracy = accuracy_score(actual_labels, predictions)
    
    assert accuracy > 0.95


class ClassifyDigits:
    def __call__(self, images: np.ndarray) -> np.ndarray[np.int_]:
        """Classify the input digits using a trained model."""
        
        normalized_images = images / 255.0
        flattened_images = normalized_images.reshape(-1, 28 * 28)
        
        predictions = tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH).predict(flattened_images)
        
        return np.array([int(np.argmax(prediction)) for prediction in predictions])
