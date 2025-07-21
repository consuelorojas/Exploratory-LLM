import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))
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

    # Get the test set and its corresponding labels
    images, expected_labels = test_set
    
    # Preprocess the images to match the input format required by the model
    preprocessed_images = np.array([image / 255.0 for image in images])
    
    # Use the classifier instance to classify the test set
    predicted_labels = classifier(preprocessed_images.reshape(-1, *images.shape[1:]))
    
    # Calculate accuracy based on correct predictions and total number of samples
    num_correct_predictions = np.sum(predicted_labels == expected_labels)
    accuracy = (num_correct_predictions / len(expected_labels)) * 100
    
    assert accuracy > 95

class ClassifyDigits(IClassifyDigits):
    def __call__(self, images: np.ndarray) -> np.ndarray:
        predictions = model.predict(images.reshape(-1, 28*28))
        return np.array([int(np.argmax(prediction)) for prediction in predictions])
