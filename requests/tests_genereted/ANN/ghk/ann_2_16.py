import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))


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
    """Load the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Generate a sample test set of images and their corresponding labels."""
    # For demonstration purposes, we'll use 10 random images with known labels.
    num_images = 10
    image_size = (28, 28)
    
    images = np.random.rand(num_images, *image_size).astype(np.uint8)  # Generate random pixel values for each image
    
    # Assign a label to each image. For simplicity, we'll use the index of the image as its label.
    labels = np.arange(0, num_images)

    return images, labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test that the digit recognition model achieves an accuracy of more than 95% on a given test set."""
    
    # Create an instance of IClassifyDigits
    classifier = ClassifyDigits()
    
    # Extract images and labels from the test set fixture.
    images, expected_labels = test_set
    
    # Use the classifier to predict labels for each image in the test set.
    predicted_labels = classifier(images)
    
    # Calculate accuracy by comparing predicted labels with actual labels
    correct_predictions = np.sum(predicted_labels == expected_labels)
    total_images = len(expected_labels)
    accuracy = (correct_predictions / total_images) * 100
    
    assert accuracy > 95, f"Expected an accuracy of more than 95%, but got {accuracy:.2f}%"

class ClassifyDigits(IClassifyDigits):
    def __call__(self, images: np.ndarray) -> np.ndarray:
        """Classifies a set of digit images using the provided model."""
        
        # Normalize pixel values to be between 0 and 1
        normalized_images = images / 255.0
        
        # Reshape each image into a flat array for input into the neural network.
        flattened_images = normalized_images.reshape(-1, 28 * 28)
        
        predictions = model.predict(flattened_images)
        
        return np.array([int(np.argmax(prediction)) for prediction in predictions])
