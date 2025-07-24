import tensorflow as tf
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
    """Fixture to load the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Fixture to generate a sample test set of images."""
    # Generate some random 28x28 grayscale images for testing purposes.
    num_images = 1000
    images = np.random.randint(0, 256, size=(num_images, 28, 28), dtype=np.uint8)
    
    return images

@pytest.fixture
def classifier():
    """Fixture to create an instance of the ClassifyDigits class."""
    from digits_classifier import ClassifyDigits
    
    return ClassifyDigits()

def test_digit_recognition_accuracy(model: load_model, test_set: np.ndarray, classifier: IClassifyDigits):
    """
    Test that the digit recognition model achieves more than 95% accuracy on a sample test set.
    
    Given:
        - A trained digit recognition model
        - A test set of images
    
    When:
        - The test set is classified using the model
    
    Then:
        - An accuracy of more than 95 percent is achieved
    """
    # Generate some random labels for testing purposes (this should be replaced with actual labels).
    num_images = len(test_set)
    labels = np.random.randint(0, 10, size=num_images)

    predictions = classifier(images=test_set / 255.0)  # Normalize the images
    
    accuracy = sum(predictions == labels) / num_images
    assert accuracy > 0.95

def test_digit_recognition_model_loads_correctly(model: load_model):
    """
    Test that the digit recognition model loads correctly.
    
    Given:
        - A trained digit recognition model file path
    
    When:
        - The model is loaded using tensorflow.keras.models.load_model()
    
    Then:
        - No exceptions are raised during loading
    """
    assert isinstance(model, tf.keras.Model)

def test_digit_recognition_classifier_initializes_correctly(classifier: IClassifyDigits):
    """
    Test that the ClassifyDigits class initializes correctly.
    
    Given:
        - A valid instance of the ClassifyDigits class
    
    When:
        - The classifier is created using its constructor
    
    Then:
        - No exceptions are raised during initialization
    """
    assert isinstance(classifier, IClassifyDigits)
