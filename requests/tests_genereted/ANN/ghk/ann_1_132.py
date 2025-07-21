import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))
from main import ClassifyDigits


# tests/test_digit_recognition.py

import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

@pytest.fixture
def model():
    """Loads the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Generates a sample test set of images with known labels."""
    # For demonstration purposes, we'll use a simple dataset.
    # In practice, you would replace this with your actual test data loading logic.
    num_samples = 10
    images = np.random.rand(num_samples, 28, 28)
    labels = np.random.randint(0, 10, size=num_samples)

    return images, labels

@pytest.fixture
def classifier():
    """Creates an instance of the digit classification interface."""
    class ClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            # Normalize and flatten the input images.
            images = images / 255.0
            images = images.reshape(-1, 28 * 28)

            predictions = model.predict(images)
            return np.array([int(np.argmax(prediction)) for prediction in predictions])

    return ClassifyDigits()

def test_digit_recognition_accuracy(model: load_model, classifier: IClassifyDigits, test_set):
    """Tests the accuracy of the digit recognition model."""
    images, labels = test_set

    # Save each image to a temporary file.
    temp_files = []
    for i in range(len(images)):
        img_path = f"temp_{i}.png"
        Image.fromarray((images[i] * 255).astype(np.uint8)).save(img_path)
        temp_files.append(img_path)

    # Load the images back into numpy arrays and classify them.
    classified_images = np.array([np.array(Image.open(file).convert('L').resize((28, 28))) for file in temp_files])
    predictions = classifier(classified_images)

    # Remove temporary files
    for file in temp_files:
        os.remove(file)

    accuracy = sum(predictions == labels) / len(labels)
    assert accuracy > 0.95

