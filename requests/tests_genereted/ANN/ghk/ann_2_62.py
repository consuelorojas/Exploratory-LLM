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

@pytest.fixture
def model():
    """Fixture to load the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Fixture to generate a sample test set of images and their corresponding labels."""
    # Generate some random data for testing purposes.
    np.random.seed(0)
    num_samples = 100
    image_size = (28, 28)
    
    images = np.random.randint(low=0, high=256, size=(num_samples,) + image_size).astype(np.uint8)
    labels = np.random.randint(low=0, high=10, size=num_samples)

    return images, labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test the accuracy of the digit recognition model on a sample test set."""
    
    # Load the ClassifyDigits class
    from digits_classifier import ClassifyDigits
    
    classifier = ClassifyDigits()
    
    # Get the images and labels from the test set fixture.
    images, expected_labels = test_set

    # Convert the images to grayscale (L mode) as required by the model.
    gray_images = np.array([np.array(Image.fromarray(image).convert('L')) for image in images])

    # Classify each digit using the trained model
    predicted_labels = classifier(gray_images)

    # Calculate accuracy of predictions against expected labels
    correct_predictions = sum(1 for pred, exp in zip(predicted_labels, expected_labels) if pred == exp)
    accuracy = (correct_predictions / len(expected_labels)) * 100

    assert accuracy > 95.0, f"Expected an accuracy greater than or equal to 95%, but got {accuracy:.2f}"
