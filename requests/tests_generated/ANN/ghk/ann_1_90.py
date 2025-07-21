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
    # For demonstration, we'll use random data. In real-world scenarios,
    # you would load your actual test dataset here.
    images = np.random.rand(100, 28, 28)
    labels = np.random.randint(0, 10, size=100)
    return images, labels

@pytest.fixture
def classifier(model):
    """Create an instance of the digit classification class."""
    class ClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            images = images / 255.0                 # normalize
            images = images.reshape(-1, 28 * 28)    # flatten

            predictions = model.predict(images)
            return np.array([int(np.argmax(prediction)) for prediction in predictions])

    return ClassifyDigits()

def test_digit_recognition_accuracy(classifier: IClassifyDigits, test_set):
    """Test the accuracy of digit recognition."""
    images, labels = test_set
    # Convert to grayscale and resize if necessary (for demonstration purposes)
    resized_images = np.array([np.array(Image.fromarray(image).convert('L').resize((28, 28))) for image in images])
    
    predictions = classifier(resized_images)

    accuracy = sum(predictions == labels) / len(labels)
    assert accuracy > 0.95
