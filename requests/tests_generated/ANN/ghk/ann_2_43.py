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
    labels = np.arange(0, num_images)
    
    return images, labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test that the digit recognition model achieves an accuracy of more than 95% on a given test set."""
    # Create an instance of IClassifyDigits
    classifier = ClassifyDigits()
    
    # Extract the images and labels from the test set fixture.
    images, expected_labels = test_set
    
    # Use the classifier to predict the labels for each image in the test set.
    predicted_labels = np.array([int(np.argmax(classifier(images=np.expand_dims(image, axis=0)))) for image in images])
    
    # Calculate the accuracy of the model on this test set by comparing its predictions with the expected labels.
    accuracy = sum(predicted_labels == expected_labels) / len(expected_labels)
    
    assert accuracy > 0.95
