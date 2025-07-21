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
    """Generate a sample test set for digit classification."""
    # For demonstration purposes, we'll use random images.
    # In practice, you should replace this with your actual test data.
    num_samples = 1000
    image_size = (28, 28)
    labels = np.random.randint(10, size=num_samples)

    def generate_random_image(label):
        img_array = np.zeros((image_size[1], image_size[0]), dtype=np.uint8) + label * 25.5
        return Image.fromarray(img_array).convert('L')

    images = [generate_random_image(label) for label in labels]
    test_images = np.array([np.array(image.resize((28, 28))) / 255.0 for image in images])
    return test_images.reshape(-1, 28 * 28), labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Test the accuracy of digit recognition using a trained model."""
    classifier = IClassifyDigits()
    predictions = np.array([int(np.argmax(prediction)) for prediction in model.predict(test_set[0].reshape(-1, 28 * 28)))])
    
    # Calculate accuracy
    correct_predictions = sum(predictions == test_set[1])
    accuracy = (correct_predictions / len(test_set[1])) * 100

    assert accuracy > 95.0
