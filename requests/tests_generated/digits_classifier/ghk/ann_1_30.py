import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






# tests/test_digit_recognition.py

import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
from tensorflow.keras.models import load_model
from PIL.Image import open, Image
import os


@pytest.fixture
def model():
    """Load the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generate a sample test set of images with known labels."""
    # For simplicity, let's assume we have 10 images in our test set.
    num_images = 10

    # Generate random pixel values for each image (28x28 grayscale).
    images = np.random.randint(0, 256, size=(num_images, 28, 28), dtype=np.uint8)

    # Assign labels to the generated images. For simplicity, let's assume
    # we have one label per digit from 0 to num_images - 1.
    labels = np.arange(num_images) % 10

    return images, labels


def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test the accuracy of the trained model on a sample test set."""
    # Load and initialize our classifier with the given model
    class ClassifyDigits(IClassifyDigits):
        def __init__(self, model: tf.keras.Model):
            self.model = model

        def __call__(self, images: np.ndarray) -> np.ndarray:
            """Classify a batch of digit images."""
            # Normalize and flatten input data.
            normalized_images = (images / 255.0).reshape(-1, 28 * 28)

            predictions = self.model.predict(normalized_images)
            return np.array([int(np.argmax(prediction)) for prediction in predictions])

    classifier: IClassifyDigits = ClassifyDigits(model=model)

    # Get the test set images and labels.
    images, expected_labels = test_set

    # Use our classifier to predict labels for each image in the test set.
    predicted_labels = classifier(images=images)

    # Calculate accuracy based on correct predictions vs total number of samples
    num_correct_predictions: int = np.sum(predicted_labels == expected_labels)
    accuracy: float = (num_correct_predictions / len(test_set[0])) * 100

    assert (
        accuracy > 95.0
    ), f"Model did not meet the required accuracy threshold ({accuracy:.2f}%)"


if __name__ == "__main__":
    pytest.main([os.path.basename(__file__)])
