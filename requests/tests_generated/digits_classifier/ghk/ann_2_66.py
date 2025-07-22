import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






# tests/test_digit_recognition.py

import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image


@pytest.fixture
def model():
    """Load the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generate a sample test set for demonstration purposes.
    
    In a real-world scenario, this would be replaced with an actual test dataset.
    """
    # Generate some random images and their corresponding labels
    np.random.seed(0)
    num_samples = 100
    image_size = (28, 28)
    images = np.random.randint(0, 256, size=(num_samples, *image_size), dtype=np.uint8)
    labels = np.random.randint(0, 10, size=num_samples)

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

    # Get the test set and labels
    images, labels = test_set
    
    # Convert to grayscale (as required by the model)
    gray_images = [Image.fromarray(image).convert('L') for image in images]
    
    # Resize images to 28x28 pixels as expected by the classifier
    resized_gray_images = np.array([np.array(img.resize((28, 28))) for img in gray_images])

    # Classify the test set using the model
    predictions = classifier(resized_gray_images)

    # Calculate accuracy
    correct_predictions = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    accuracy = (correct_predictions / len(labels)) * 100

    assert accuracy > 95.0


def test_digit_recognition_interface():
    """Test the IClassifyDigits interface."""
    
    class ClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            return np.array([1] * len(images))

    classifier = ClassifyDigits()
    assert isinstance(classifier(np.random.rand(10, 28, 28)), np.ndarray)
