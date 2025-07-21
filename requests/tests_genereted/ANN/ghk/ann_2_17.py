import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))
from main import ClassifyDigits


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
    num_samples = 1000
    image_size = (28, 28)
    
    # Generate dummy data with labels
    x_test = np.random.rand(num_samples, *image_size).astype(np.float32) / 255.0
    y_test = np.random.randint(10, size=num_samples)

    return x_test.reshape(-1, 28*28), y_test

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Tests the accuracy of the digit recognition model."""
    
    # Create an instance of IClassifyDigits
    classifier = ClassifyDigits()
    
    # Get predictions from our test set using the loaded model and classifier
    x_test, y_test = test_set
    
    # Use the classifier to get predicted labels for our test data
    predicted_labels = classifier(x=x_test)
    
    # Calculate accuracy by comparing actual vs. predicted labels
    correct_predictions = np.sum(predicted_labels == y_test)
    accuracy = (correct_predictions / len(y_test)) * 100

    assert accuracy > 95, f"Expected model to have an accuracy of more than 95%, but got {accuracy:.2f}%"
