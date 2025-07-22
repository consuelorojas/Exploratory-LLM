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
    """Loads the trained digit recognition model."""
    return tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Generates a sample test set for testing purposes."""
    # For demonstration, we'll use random images. In real-world scenarios,
    # you would load your actual test dataset here.
    num_images = 1000
    image_size = (28, 28)
    
    images = np.random.rand(num_images, *image_size).astype(np.uint8)
    labels = np.random.randint(10, size=num_images)

    return images, labels

@pytest.fixture
def classifier(model):
    """Creates an instance of the digit classification class."""
    from digits_classifier import ClassifyDigits  # Importing locally to avoid circular imports
    
    return ClassifyDigits()

def test_digit_recognition_accuracy(classifier: IClassifyDigits, model, test_set):
    """
    Tests if the trained digit recognition model achieves more than 95% accuracy on a given test set.
    
    :param classifier: An instance of the class that performs digit classification
    :param model: The loaded TensorFlow model for digit recognition (not directly used in this function)
    :param test_set: A tuple containing images and their corresponding labels
    """
    # Extracting images and labels from the test set fixture
    images, expected_labels = test_set
    
    # Normalizing images to match what's done within ClassifyDigits
    normalized_images = (images / 255.0).reshape(-1, 28 * 28)
    
    # Predictions are made using the classifier instance which encapsulates model usage
    predicted_labels = np.array([int(np.argmax(prediction)) for prediction in classifier(normalized_images)])
    
    accuracy = sum(predicted_labels == expected_labels) / len(expected_labels)
    
    assert accuracy > 0.95, f"Expected an accuracy of more than 95%, but got {accuracy}"
