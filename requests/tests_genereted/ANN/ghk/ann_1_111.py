import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))


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
    
    # Generate some dummy data. In real-world scenarios, you'd load your actual dataset here.
    images = np.random.rand(num_images, *image_size).astype(np.uint8) 
    labels = np.arange(0, num_images)

    return {"images": images, "labels": labels}

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Tests the accuracy of digit recognition model on a given test set."""
    
    # Initialize our classifier
    classify_digits = ClassifyDigits()
    
    # Normalize and flatten input data for prediction.
    normalized_images = (test_set["images"] / 255.0).reshape(-1, 28 * 28)
    
    predictions = model.predict(normalized_images)
    predicted_labels = np.array([int(np.argmax(prediction)) for prediction in predictions])
    
    accuracy = sum(predicted_labels == test_set['labels']) / len(test_set['labels'])
    
    assert accuracy > 0.95
