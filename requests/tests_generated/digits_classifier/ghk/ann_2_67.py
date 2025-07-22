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
    """Loads the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generates a sample test set of images for classification."""
    # For demonstration purposes, we'll use a single image.
    # In a real-world scenario, you'd want to generate or load multiple images here.
    img_path = os.path.join(os.getcwd(), "tests", "test_image.png")
    return np.array(open(img_path).convert('L').resize((28, 28)))


@pytest.fixture
def classifier():
    """Initializes the digit classification class."""
    from digits_classifier import ClassifyDigits
    return ClassifyDigits()


def test_digit_recognition_accuracy(model: load_model, test_set: np.ndarray, classifier: IClassifyDigits):
    """
    Tests if the trained model achieves an accuracy of more than 95% on a given test set.

    Args:
        model (load_model): The loaded digit recognition model.
        test_set (np.ndarray): A sample image for classification.
        classifier (IClassifyDigits): An instance of the ClassifyDigits class.
    """
    # Generate predictions using the provided classifier
    prediction = classifier(test_set)

    # For demonstration purposes, assume we have a ground truth label for our single test image
    ground_truth_label = 5

    # Calculate accuracy based on this single example (in real scenarios you'd use multiple examples)
    accuracy = int(prediction == ground_truth_label) * 100 / len([prediction])

    assert accuracy > 95, f"Expected an accuracy of more than 95%, but got {accuracy}%"
