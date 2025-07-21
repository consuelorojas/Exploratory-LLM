import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))


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
    # For demonstration purposes, we'll use a simple dataset with 10 images of digits from 0 to 9.
    # In practice, you should replace this with your actual test data.
    images = []
    labels = []
    
    for i in range(10):
        image_path = f"tests/test_images/{i}.png"
        if os.path.exists(image_path):
            img = np.array(Image.open(image_path).convert('L').resize((28, 28)))
            images.append(img)
            labels.append(i)

    return np.array(images), np.array(labels)

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Tests the accuracy of digit recognition using a trained model."""
    
    # Initialize the classifier
    class ClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            images = images / 255.0                 # normalize
            images = images.reshape(-1, 28 * 28)    # flatten

            predictions = model.predict(images)
            return np.array([int(np.argmax(prediction)) for prediction in predictions])

    classifier = ClassifyDigits()
    
    # Get the test set and labels
    images, labels = test_set
    
    # Make predictions using the trained model
    predicted_labels = classifier(images)

    # Calculate accuracy
    correct_predictions = sum(predicted_labels == labels)
    total_images = len(labels)
    accuracy = (correct_predictions / total_images) * 100

    assert accuracy > 95, f"Accuracy {accuracy} is less than expected threshold of 95%"
