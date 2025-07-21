import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))
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
    
    # Load and initialize the ClassifyDigits class
    classifier = IClassifyDigits()
    
    # Extract images and labels from the test set fixture
    images, expected_labels = test_set
    
    # Convert PIL Image to numpy array for each image in the test set.
    np_images = [np.array(Image.fromarray(image).convert('L').resize((28, 28))) for image in images]
    
    # Classify the digits using the model and classifier
    predicted_labels = classifier(np.array(np_images))
    
    # Calculate accuracy by comparing expected labels with predicted labels
    correct_predictions = np.sum(expected_labels == predicted_labels)
    total_samples = len(test_set[0])
    accuracy = (correct_predictions / total_samples) * 100
    
    assert accuracy > 95, f"Expected an accuracy of more than 95%, but got {accuracy:.2f}%"
