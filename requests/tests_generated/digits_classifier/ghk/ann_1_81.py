import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






# tests/test_digit_recognition.py

import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
from PIL import Image
import tensorflow as tf

@pytest.fixture
def model():
    """Load the trained digit recognition model."""
    return tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Generate a sample test set for testing purposes."""
    # For demonstration, we'll use 10 random images with known labels.
    np.random.seed(0)
    images = np.random.rand(10, 28, 28) * 255.0
    labels = np.random.randint(0, 9, size=10)

    return images.astype(np.uint8), labels

@pytest.fixture
def classifier(model):
    """Create an instance of the digit classification class."""
    from digits_classifier import ClassifyDigits
    return ClassifyDigits()

def test_digit_recognition_accuracy(classifier: IClassifyDigits, model, test_set):
    """
    Test that the trained digit recognition model achieves more than 95% accuracy on a sample test set.
    
    Given:
        - A trained digit recognition model (model)
        - A test set of images with known labels (test_set)
        
    When:
        - The test set is classified using the provided model
        
    Then:
        - An accuracy of more than 95 percent should be achieved
    """
    # Extract the images and their corresponding labels from the test set.
    images, expected_labels = test_set

    # Classify each image in the test set.
    predicted_labels = classifier(images)

    # Calculate the number of correct predictions (accuracy).
    accuracy = np.sum(predicted_labels == expected_labels) / len(expected_labels)
    
    assert accuracy > 0.95
