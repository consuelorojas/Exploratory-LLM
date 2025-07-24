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
    """Loads the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Generates a sample test set for testing purposes."""
    # For demonstration, we'll use 10 random images with known labels.
    num_images = 10
    image_size = (28, 28)
    
    # Generate some dummy data. Replace this with actual test data in your project structure.
    images = np.random.rand(num_images, *image_size).astype(np.uint8) 
    labels = np.arange(0, num_images)

    return images, labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Tests the accuracy of digit recognition using a trained model."""
    
    # Load ClassifyDigits class
    from digits_classifier import ClassifyDigits
    
    classifier = ClassifyDigits()
    
    # Extract test set data and labels.
    images, expected_labels = test_set

    # Normalize input to match what our classify_digits function expects.
    normalized_images = np.array([img / 255.0 for img in images])
    flattened_images = normalized_images.reshape(-1, 28 * 28)

    predictions = model.predict(flattened_images)
    
    predicted_labels = [int(np.argmax(prediction)) % len(expected_labels) for prediction in predictions]
    
    accuracy = np.mean([predicted == expected for predicted, expected in zip(predicted_labels, expected_labels)])
    
    assert accuracy > 0.95
