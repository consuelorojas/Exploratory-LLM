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
from PIL.Image import open, Image
import os


@pytest.fixture
def model():
    """Loads the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generates a sample test set of images with known labels."""
    # For demonstration purposes, we'll use 10 random images.
    num_images = 10
    image_size = (28, 28)
    
    # Generate some dummy data for testing. In real-world scenarios,
    # you would load your actual test dataset here.
    images = np.random.rand(num_images, *image_size).astype(np.uint8)
    labels = np.random.randint(0, 10, num_images)

    return images, labels


def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Tests the accuracy of the digit recognition model."""
    
    # Create an instance of IClassifyDigits
    classifier = ClassifyDigits()
    
    # Extract the images and labels from the test set.
    images, expected_labels = test_set
    
    # Convert PIL Image to numpy array for each image in the test set
    np_images = [np.array(image) if isinstance(image, Image.Image) else image 
                 for image in images]
    
    # Classify the test set using the model.
    predicted_labels = classifier(np.array(np_images))
    
    # Calculate accuracy by comparing expected labels with predicted ones.
    correct_predictions = sum(1 for pred, exp in zip(predicted_labels, expected_labels)
                              if pred == exp)
    accuracy = (correct_predictions / len(expected_labels)) * 100
    
    assert accuracy > 95
