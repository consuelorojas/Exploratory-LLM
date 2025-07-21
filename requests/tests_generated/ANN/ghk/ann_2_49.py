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
    """Generate a sample test set of images and their corresponding labels."""
    # For demonstration purposes, we'll use 10 random images with known labels.
    num_images = 10
    image_size = (28, 28)
    
    images = np.random.rand(num_images, *image_size).astype(np.uint8)  # Generate random pixel values for each image
    
    # Assign a label to each image. In this case, we'll use the index of the image as its label.
    labels = np.arange(num_images)

    return images, labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test that the digit recognition model achieves an accuracy of more than 95% on a given test set."""
    
    # Create an instance of IClassifyDigits
    classifier = ClassifyDigits()
    
    # Extract images and labels from the test set fixture.
    images, expected_labels = test_set
    
    # Convert images to grayscale (if not already) and resize them if necessary.
    resized_images = np.array([Image.fromarray(image).convert('L').resize((28, 28)) for image in images])
    
    # Classify each image using the model
    predicted_labels = classifier(resized_images)
    
    # Calculate accuracy by comparing expected labels with predicted ones
    correct_predictions = sum(1 for pred_label, exp_label in zip(predicted_labels, expected_labels) if pred_label == exp_label % 10)  # Ensure label comparison is within digit range (0-9).
    accuracy = correct_predictions / len(expected_labels)
    
    assert accuracy > 0.95
