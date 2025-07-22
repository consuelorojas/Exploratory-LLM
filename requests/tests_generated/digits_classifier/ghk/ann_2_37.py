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
    """Loads the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Generates a sample test set of images with known labels."""
    # For demonstration purposes, we'll use 10 random images.
    num_images = 10
    image_size = (28, 28)
    
    # Generate some dummy data for testing. In real-world scenarios,
    # you would load your actual dataset here.
    test_set_images = np.random.rand(num_images, *image_size).astype(np.uint8)
    test_set_labels = np.random.randint(0, 10, num_images)

    return test_set_images, test_set_labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Tests the accuracy of the digit recognition model."""
    
    # Extract images and labels from the test set
    test_set_images, test_set_labels = test_set
    
    # Create an instance of IClassifyDigits to classify digits using our model.
    classifier = ClassifyDigits()
    
    # Normalize and flatten the input data for classification
    normalized_test_set_images = (test_set_images / 255.0).reshape(-1, 28 * 28)
    
    # Use the loaded model directly in this test case instead of relying on 
    # IClassifyDigits to load it again.
    predictions = model.predict(normalized_test_set_images)
    predicted_labels = np.array([int(np.argmax(prediction)) for prediction in predictions])
    
    accuracy = sum(predicted_labels == test_set_labels) / len(test_set_labels)

    assert accuracy > 0.95, f"Expected an accuracy of more than 95%, but got {accuracy:.2f}"
