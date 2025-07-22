import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






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


@pytest.fixture
def classifier(model):
    """Create an instance of the digit classification class."""
    from digits_classifier import ClassifyDigits
    
    return ClassifyDigits()


def test_digit_recognition_accuracy(classifier: IClassifyDigits, model, test_set):
    """
    Test that the trained digit recognition model achieves more than 95% accuracy on a given test set.
    
    Given:
        - A trained digit recognition model
        - A test set of images with known labels
    
    When:
        - The test set is classified using the model
    
    Then:
        - An accuracy of more than 95 percent should be achieved
    """
    # Extract the test set and its corresponding labels.
    test_set_images, expected_labels = test_set

    # Classify each image in the test set.
    predicted_labels = classifier(test_set_images)

    # Calculate the accuracy by comparing the predicted labels with the actual ones.
    correct_predictions = np.sum(predicted_labels == expected_labels)
    accuracy = (correct_predictions / len(expected_labels)) * 100
    
    assert accuracy > 95, f"Expected an accuracy of more than 95%, but got {accuracy:.2f}%"
