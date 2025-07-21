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
    # For demonstration purposes, we'll use a simple dataset with 10 images.
    # In practice, you'd want to replace this with your actual test data.
    num_images = 10
    image_size = (28, 28)
    
    images = np.random.rand(num_images, *image_size).astype(np.uint8)  # Random pixel values for demonstration purposes
    
    labels = np.arange(0, num_images % 10)  # Simple label assignment: each image is labeled with its index modulo 10
    return images, labels

def test_digit_recognition_accuracy(model, test_set):
    """Test the accuracy of digit recognition using a trained model."""
    
    classifier = IClassifyDigits()
    images, expected_labels = test_set
    
    # Preprocess the images (resize and normalize)
    preprocessed_images = np.array([Image.fromarray(image).convert('L').resize((28, 28)) for image in images]) / 255.0
    predicted_labels = classifier(preprocessed_images.reshape(-1, 28 * 28))
    
    # Calculate accuracy: proportion of correctly classified digits
    correct_classifications = np.sum(predicted_labels == expected_labels)
    accuracy = (correct_classifications / len(expected_labels)) * 100
    
    assert accuracy > 95.0

