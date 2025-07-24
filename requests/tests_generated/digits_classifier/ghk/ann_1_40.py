import tensorflow as tf
import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






# tests/test_digit_recognition.py

import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
from tensorflow.keras.models import load_model
from PIL.Image import open, Image


@pytest.fixture
def model():
    """Load the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generate a sample test set for demonstration purposes.
    
    In a real-world scenario, this would be replaced with an actual test dataset.
    """
    # Generate random images and labels (this is just a placeholder)
    num_samples = 1000
    image_size = (28, 28)
    images = np.random.rand(num_samples, *image_size).astype(np.uint8)
    labels = np.random.randint(10, size=num_samples)

    return images, labels


def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test the accuracy of the digit recognition model."""
    
    # Load the classifier
    class ClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            images = images / 255.0                 # normalize
            images = images.reshape(-1, 28 * 28)    # flatten

            predictions = model.predict(images)
            return np.array([int(np.argmax(prediction)) for prediction in predictions])

    classifier = ClassifyDigits()

    # Get the test set
    images, labels = test_set
    
    # Convert to grayscale and resize if necessary (assuming they are already 28x28)
    gray_images = [np.array(open(Image.fromarray(image).convert('L'))) for image in images]

    # Make predictions on the test set
    predicted_labels = classifier(np.stack(gray_images))

    # Calculate accuracy
    correct_predictions = np.sum(predicted_labels == labels)
    accuracy = (correct_predictions / len(labels)) * 100

    assert accuracy > 95, f"Accuracy {accuracy:.2f} is less than expected"
