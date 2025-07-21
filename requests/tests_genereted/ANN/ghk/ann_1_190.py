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

@pytest.fixture
def model():
    """Load the trained digit recognition model."""
    return tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Generate a sample test set for testing purposes."""
    # For demonstration, we'll use 10 random images with known labels.
    np.random.seed(0)
    num_samples = 100
    image_size = (28, 28)
    images = np.random.rand(num_samples, *image_size).astype(np.uint8) / 255.0
    labels = np.random.randint(low=0, high=10, size=num_samples)

    return images, labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test the accuracy of the digit recognition model."""
    # Create an instance of IClassifyDigits to classify digits.
    classifier = ClassifyDigits()

    # Extract the test set and its corresponding labels.
    images, expected_labels = test_set

    # Convert PIL Image objects if necessary
    pil_images = [Image.fromarray(image * 255).convert('L').resize((28, 28)) for image in images]

    predicted_labels = classifier(images=np.array(pil_images))

    accuracy = np.mean(predicted_labels == expected_labels)

    assert accuracy > 0.95

class ClassifyDigits(IClassifyDigits):
    def __call__(self, images: np.ndarray) -> np.ndarray:
        """Classify the given set of digit images."""
        # Normalize and flatten the input data.
        normalized_images = images / 255.0
        flattened_images = normalized_images.reshape(-1, 28 * 28)

        predictions = model.predict(flattened_images)
        return np.array([int(np.argmax(prediction)) for prediction in predictions])
