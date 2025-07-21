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
    labels = np.random.randint(0, 9, num_images)

    return images, labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Tests the accuracy of the digit recognition model."""
    
    # Load and normalize the data
    images, expected_labels = test_set
    
    classifier = IClassifyDigits()
    predicted_labels = np.array([int(np.argmax(prediction)) for prediction in 
                                 load_model(MODEL_DIGIT_RECOGNITION_PATH).predict(images / 255.0.reshape(-1, 28 * 28))])
    
    # Calculate accuracy
    correct_predictions = sum(1 for pred_label, exp_label in zip(predicted_labels, expected_labels) if pred_label == exp_label)
    accuracy = (correct_predictions / len(expected_labels)) * 100
    
    assert accuracy > 95

def test_classify_digits_interface():
    """Tests the ClassifyDigits interface."""
    
    # Create an instance of IClassifyDigits
    classifier: IClassifyDigits = ClassifyDigits()
    
    # Generate some dummy data. In real-world scenarios, you'd load your actual dataset here.
    num_images = 10
    image_size = (28, 28)
    images = np.random.rand(num_images, *image_size).astype(np.uint8) 
    
    predicted_labels: NDArray[np.int_] = classifier(images=images)

    assert len(predicted_labels.shape) == 1 and predicted_labels.dtype == np.int_
