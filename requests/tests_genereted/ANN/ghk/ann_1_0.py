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
from sklearn.metrics import accuracy_score


@pytest.fixture
def model():
    """Load the trained digit recognition model."""
    return tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generate a sample test set for demonstration purposes.
    
    In a real-world scenario, this would be replaced with an actual test dataset.
    """
    # Generate some random images and labels (this is just a placeholder)
    np.random.seed(0)  # For reproducibility
    num_samples = 100
    image_size = (28, 28)
    
    images = np.random.randint(low=0, high=256, size=(num_samples,) + image_size).astype(np.uint8)
    labels = np.random.randint(low=0, high=10, size=num_samples)

    return images, labels


@pytest.fixture
def classifier(model):
    """Create an instance of the digit classification class."""
    from digits_classifier import ClassifyDigits  # Importing locally to avoid circular imports
    
    return ClassifyDigits()


def test_digit_recognition_accuracy(classifier: IClassifyDigits, model, test_set):
    """
    Test that the trained digit recognition model achieves more than 95% accuracy on a given test set.
    
    Given:
        - A trained digit recognition model
        - A test set of images and their corresponding labels
    
    When:
        - The test set is classified using the provided model
    
    Then:
        - An accuracy of more than 95 percent should be achieved
    """
    # Extract the test data from the fixture
    images, expected_labels = test_set

    # Normalize and flatten the input images as required by the classifier
    normalized_images = (images / 255.0).reshape(-1, 28 * 28)

    # Use the model to predict labels for the given images
    predicted_labels = np.array([int(np.argmax(model.predict(image.reshape(1, -1)))) for image in normalized_images])

    # Calculate and assert accuracy
    accuracy = accuracy_score(expected_labels, predicted_labels)
    
    assert accuracy > 0.95, f"Expected an accuracy of more than 95%, but got {accuracy:.2f}"
