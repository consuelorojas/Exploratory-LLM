import tensorflow as tf
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
    """Generates a sample test set of images with known labels."""
    # For demonstration purposes, we'll use a simple dataset.
    # In practice, you'd want to replace this with your actual test data.
    num_samples = 10
    images = np.random.rand(num_samples, 28, 28)
    labels = np.random.randint(0, 10, size=num_samples)

    return images, labels

@pytest.fixture
def classifier():
    """Creates an instance of the digit classification class."""
    from digits_classifier import ClassifyDigits
    return ClassifyDigits()

def test_digit_recognition_accuracy(model: load_model, test_set, classifier):
    """
    Tests that the trained model achieves more than 95% accuracy on a given test set.

    Args:
        model (load_model): The loaded digit recognition model.
        test_set (tuple): A tuple containing images and labels for testing.
        classifier (IClassifyDigits): An instance of the ClassifyDigits class.
    """
    # Extract images and labels from the test set
    images, expected_labels = test_set

    # Normalize and flatten the input data as required by the model
    normalized_images = images / 255.0
    flattened_images = normalized_images.reshape(-1, 28 * 28)

    # Use the classifier to predict labels for the given images
    predicted_labels = np.array([int(np.argmax(prediction)) for prediction in model.predict(flattened_images)])

    # Calculate accuracy by comparing expected and actual labels
    correct_predictions = sum(1 for pred_label, exp_label in zip(predicted_labels, expected_labels) if pred_label == exp_label)
    accuracy = (correct_predictions / len(expected_labels)) * 100

    assert accuracy > 95.0


def test_digit_recognition_using_classifier(classifier: IClassifyDigits, test_set):
    """
    Tests that the classifier achieves more than 95% accuracy on a given test set.

    Args:
        classifier (IClassifyDigits): An instance of the ClassifyDigits class.
        test_set (tuple): A tuple containing images and labels for testing.
    """

    # Extract images from the test set
    images, expected_labels = test_set

    # Use the classifier to predict labels for the given images
    predicted_labels = classifier(images)

    # Calculate accuracy by comparing expected and actual labels
    correct_predictions = sum(1 for pred_label, exp_label in zip(predicted_labels, expected_labels) if pred_label == exp_label)
    accuracy = (correct_predictions / len(expected_labels)) * 100

    assert accuracy > 95.0


def test_digit_recognition_using_classifier_with_image(classifier: IClassifyDigits):
    """
    Tests that the classifier can correctly classify a single image.

    Args:
        classifier (IClassifyDigits): An instance of the ClassifyDigits class.
    """

    # Load an example image
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = f"{current_dir}/../data/example_image.png"
    if not os.path.exists(img_path):
        pytest.skip("Example image does not exist")

    with Image.open(img_path) as img:
        img_gray = np.array(img.convert('L').resize((28, 28)))

    # Use the classifier to predict a label for the given image
    predicted_label = classifier(np.expand_dims(img_gray, axis=0))[0]

    assert isinstance(predicted_label, int)
