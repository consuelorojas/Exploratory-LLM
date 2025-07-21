import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'ANN'))
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
    """Load the trained digit recognition model."""
    return tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Generate a sample test set for testing purposes."""
    # For demonstration, we'll use random images. In real-world scenarios,
    # you would load your actual test dataset.
    num_samples = 1000
    image_size = (28, 28)
    
    # Generate dummy data with labels from 0 to 9
    x_test = np.random.rand(num_samples, *image_size).astype(np.float32) / 255.0
    y_test = np.random.randint(10, size=num_samples)

    return x_test.reshape(-1, 28*28), y_test

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test the accuracy of digit recognition model."""
    
    # Create an instance of ClassifyDigits
    classifier = IClassifyDigits()
    
    # Get predictions from our model using the classify_digits function
    x_test, y_test = test_set
    
    images = np.reshape(x_test, (-1, 28*28))
    predicted_labels = [int(np.argmax(prediction)) for prediction in 
                        tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH).predict(images)]
    
    # Calculate accuracy
    correct_predictions = sum(1 for pred_label, actual_label in zip(predicted_labels, y_test) if pred_label == actual_label)
    accuracy = (correct_predictions / len(y_test)) * 100
    
    assert accuracy > 95

def test_classify_digits_interface():
    """Test the ClassifyDigits interface."""
    
    class TestClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            return np.array([0] * len(images))
        
    classifier = TestClassifyDigits()
    result = classifier(np.random.rand(10, 28*28).astype(np.float32))
    
    assert isinstance(result, np.ndarray)
