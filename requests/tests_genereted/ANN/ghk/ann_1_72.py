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
    # For demonstration purposes, assume we have 10 images with correct labels.
    num_images = 10
    image_size = (28, 28)
    
    # Generate random images for testing. In real-world scenarios,
    # you would load actual images from a test dataset.
    images = np.random.rand(num_images, *image_size).astype(np.uint8)
    labels = np.arange(0, num_images) % 10
    
    return images, labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test the accuracy of digit recognition using a trained model."""
    
    # Extract images and their corresponding labels from the test set.
    images, expected_labels = test_set
    
    # Create an instance of IClassifyDigits to classify digits in the test set.
    classifier = ClassifyDigits()
    
    # Convert PIL Image objects into numpy arrays for classification
    image_arrays = np.array([np.array(Image.fromarray(image).convert('L').resize((28, 28))) for image in images])
    
    # Use the model to predict labels of digits in the test set.
    predicted_labels = classifier(images=image_arrays)
    
    # Calculate accuracy by comparing expected and predicted labels.
    correct_predictions = np.sum(predicted_labels == expected_labels)
    accuracy = (correct_predictions / len(expected_labels)) * 100
    
    assert accuracy > 95, f"Expected an accuracy of more than 95%, but got {accuracy:.2f}%"
