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
    """Load the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Generate a sample test set of images with known labels."""
    # For demonstration purposes, we'll use 10 random images.
    num_images = 10
    image_size = (28, 28)
    
    # Generate some dummy data for testing. In real-world scenarios,
    # you would load your actual dataset here.
    test_set_images = np.random.rand(num_images, *image_size).astype(np.uint8)
    test_set_labels = np.random.randint(0, 10, num_images)

    return test_set_images, test_set_labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test the accuracy of digit recognition model."""
    
    # Extract images and labels from the test set
    test_set_images, expected_labels = test_set
    
    # Create an instance of ClassifyDigits to classify digits using our trained model.
    classifier = IClassifyDigits()
    
    # Convert PIL Image objects into numpy arrays for classification.
    image_arrays = np.array([np.array(Image.fromarray(image).convert('L')) for image in test_set_images])
    
    # Use the classifier instance to get predicted labels
    predicted_labels = classifier(images=image_arrays)
    
    # Calculate accuracy by comparing expected and actual labels
    correct_predictions = sum(1 for pred, exp in zip(predicted_labels, expected_labels) if pred == exp)
    accuracy = (correct_predictions / len(test_set_images)) * 100
    
    assert accuracy > 95.0

