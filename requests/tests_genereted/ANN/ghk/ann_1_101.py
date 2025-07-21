import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))


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
    """Generate a sample test set for demonstration purposes."""
    # Replace this with your actual test data loading logic.
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
    """Test the accuracy of the digit recognition model."""
    images, labels = test_set
    # Resize and convert to grayscale (if necessary) before passing to classifier.
    resized_images = np.array([np.array(Image.fromarray(image).resize((28, 28)).convert('L')) for image in images])
    
    predictions = classifier(resized_images)
    accuracy = sum(predictions == labels) / len(labels)

    assert accuracy > 0.95
