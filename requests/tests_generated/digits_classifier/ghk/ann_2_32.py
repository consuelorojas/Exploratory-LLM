import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






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
    """Generates a sample test set for digit classification."""
    # For demonstration purposes, we'll use a simple dataset.
    images = []
    labels = []

    # Assuming there's an 'images' directory with some example images.
    image_dir = os.path.join(os.getcwd(), "tests", "test_images")
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            img_path = os.path.join(image_dir, filename)
            label = int(filename.split("_")[0])  # Assuming the file name starts with a digit.
            image = np.array(Image.open(img_path).convert('L').resize((28, 28)))
            images.append(image)
            labels.append(label)

    return np.array(images), np.array(labels)

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Tests the accuracy of the digit recognition model."""
    classifier = ClassifyDigits()
    images, expected_labels = test_set

    # Normalize and flatten the input data.
    normalized_images = images / 255.0
    flattened_images = normalized_images.reshape(-1, 28 * 28)

    predictions = model.predict(flattened_images)
    predicted_labels = np.array([int(np.argmax(prediction)) for prediction in predictions])

    accuracy = sum(predicted_labels == expected_labels) / len(expected_labels)
    assert accuracy > 0.95
