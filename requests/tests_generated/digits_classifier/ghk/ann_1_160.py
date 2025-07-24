import tensorflow as tf
import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






# tests/test_digit_recognition.py

import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
from tensorflow.keras.models import load_model
import PIL.Image


@pytest.fixture
def model():
    """Load the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generate a sample test set of images and their corresponding labels."""
    # For demonstration purposes, we'll use random data.
    np.random.seed(0)
    num_samples = 100
    image_size = (28, 28)

    images = np.random.randint(0, 256, size=(num_samples, *image_size), dtype=np.uint8)
    labels = np.random.randint(0, 10, size=num_samples)

    return images, labels


def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test the accuracy of the digit recognition model."""
    # Create an instance of IClassifyDigits
    classifier = ClassifyDigits()

    # Get the test set and its corresponding labels
    images, expected_labels = test_set

    # Convert images to grayscale (L) mode as required by the classify method
    pil_images = [PIL.Image.fromarray(image).convert('L').resize((28, 28)) for image in images]

    # Classify each image using the model and get predicted labels
    predicted_labels = classifier(np.array(pil_images))

    # Calculate accuracy
    correct_predictions = np.sum(predicted_labels == expected_labels)
    accuracy = (correct_predictions / len(expected_labels)) * 100

    assert accuracy > 95, f"Expected an accuracy of more than 95%, but got {accuracy:.2f}%"


class ClassifyDigits(IClassifyDigits):
    def __call__(self, images: np.ndarray) -> np.ndarray:
        """Classify the given set of digit images using a trained model."""
        # Normalize and flatten the input data
        normalized_images = images / 255.0
        flattened_images = normalized_images.reshape(-1, 28 * 28)

        predictions = model.predict(flattened_images)
        return np.array([int(np.argmax(prediction)) for prediction in predictions])
