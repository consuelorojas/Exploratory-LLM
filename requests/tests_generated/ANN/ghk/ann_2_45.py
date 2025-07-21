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
    """Load the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Generate a sample test set for testing purposes."""
    # For demonstration, we'll use 10 random images with known labels.
    num_images = 10
    image_size = (28, 28)
    
    # Create dummy data: each pixel value is the same as its corresponding label.
    images = np.random.randint(0, 256, size=(num_images,) + image_size).astype(np.uint8)
    labels = [np.argmax(image.flatten()) % 10 for image in images]
    
    return {
        "images": np.array([Image.fromarray(img) for img in images]),
        "labels": np.array(labels),
    }

def test_digit_recognition_accuracy(model, test_set):
    """Test the accuracy of digit recognition using a trained model."""
    classifier = IClassifyDigits()
    
    # Preprocess and classify the test set.
    predictions = []
    for image in test_set["images"]:
        img_array = np.array(image.convert('L').resize((28, 28)))
        prediction = int(classifier(np.array([img_array]))[0])
        predictions.append(prediction)
        
    accuracy = sum(1 for pred, label in zip(predictions, test_set['labels']) if pred == label) / len(test_set["images"])
    
    # Assert that the model achieves an accuracy of more than 95%.
    assert accuracy > 0.95
