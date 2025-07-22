import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






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
    # In practice, you would replace this with your actual test data loading logic.
    num_images = 10
    image_size = (28, 28)
    
    images = np.random.rand(num_images, *image_size).astype(np.uint8)  # Simulated images
    labels = np.arange(0, num_images) % 10  # Corresponding simulated labels
    
    return images, labels

@pytest.fixture
def classifier(model):
    """Create an instance of the digit classification class."""
    from digits_classifier import ClassifyDigits
    return ClassifyDigits()

def test_digit_recognition_accuracy(classifier: IClassifyDigits, model, test_set):
    """
    Test that the trained digit recognition model achieves more than 95% accuracy on a given test set.
    
    Given:
        - A trained digit recognition model (fixture)
        - A sample test set of images and their corresponding labels (fixture)
        
    When:
        - The classifier is used to classify the test set
        
    Then:
        - An accuracy of more than 95% should be achieved
    """
    # Load the test data from fixture
    images, expected_labels = test_set
    
    # Classify each image in the test set using our model
    predicted_labels = np.array([int(np.argmax(model.predict(image.reshape(1, *image.shape).astype('float32') / 255.0))) for image in images])
    
    # Calculate accuracy by comparing actual labels with expected ones
    correct_predictions = sum(predicted_labels == expected_labels)
    total_images = len(images)
    accuracy = (correct_predictions / total_images) * 100
    
    assert accuracy > 95, f"Expected an accuracy of more than 95%, but got {accuracy:.2f}%"
