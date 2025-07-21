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

@pytest.fixture
def model():
    """Loads the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Generates a sample test set of images and their corresponding labels."""
    # For demonstration purposes, we'll use random data.
    np.random.seed(0)
    num_samples = 100
    image_size = (28, 28)
    
    images = np.random.randint(low=0, high=256, size=(num_samples, *image_size), dtype=np.uint8)
    labels = np.random.randint(low=0, high=10, size=num_samples)

    return images, labels

def test_digit_recognition_accuracy(model: tf.keras.Model, test_set):
    """Tests the accuracy of the digit recognition model on a given test set."""
    
    # Load and initialize the classifier
    class ClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            images = images / 255.0                 # normalize
            images = images.reshape(-1, 28 * 28)    # flatten

            predictions = model.predict(images)
            return np.array([int(np.argmax(prediction)) for prediction in predictions])

    classifier = ClassifyDigits()

    # Extract the test set and labels
    images, labels = test_set
    
    # Convert images to grayscale (as required by the model)
    gray_images = [Image.fromarray(image).convert('L') for image in images]
    
    # Resize images to match the expected input size of 28x28 pixels
    resized_gray_images = np.array([np.array(img.resize((28, 28))) for img in gray_images])
    
    # Classify the test set using the model
    predictions = classifier(resized_gray_images)
    
    # Calculate accuracy
    correct_predictions = sum(1 for prediction, label in zip(predictions, labels) if prediction == label)
    accuracy = (correct_predictions / len(labels)) * 100
    
    assert accuracy > 95.0

