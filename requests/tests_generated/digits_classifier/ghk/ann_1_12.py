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
    """Load the trained digit recognition model."""
    return tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Generate a sample test set for testing purposes."""
    # For demonstration, we'll use random images. In real-world scenarios,
    # you would load your actual test dataset here.
    num_images = 1000
    image_size = (28, 28)
    
    images = np.random.rand(num_images, *image_size).astype(np.uint8)
    labels = np.random.randint(10, size=num_images)

    return images, labels

@pytest.fixture
def classifier():
    """Create an instance of the digit classification class."""
    from digits_classifier import ClassifyDigits  # Importing locally to avoid circular imports
    
    return ClassifyDigits()

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set, classifier: IClassifyDigits):
    """
    Test that the trained model achieves more than 95% accuracy on a given test set.
    
    Given:
        - A trained digit recognition model
        - A test set of images and their corresponding labels
    
    When:
        - The test set is classified using the provided model
    
    Then:
        - An accuracy of more than 95 percent should be achieved
    """
    # Extracting images and labels from the test_set fixture
    images, expected_labels = test_set

    # Preprocess images to match what ClassifyDigits expects (grayscale, normalized)
    preprocessed_images = np.array([Image.fromarray(image).convert('L').resize((28, 28)) for image in images]) / 255.0
    
    predicted_labels = classifier(preprocessed_images)

    accuracy = sum(predicted == expected for predicted, expected in zip(predicted_labels, expected_labels)) / len(expected_labels)
    
    assert accuracy > 0.95
