import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'ANN'))
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
    """Load the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generate a sample test set of images and their corresponding labels."""
    # For demonstration purposes, we'll use 10 random images with known labels.
    num_images = 1000
    image_size = (28, 28)
    images = np.random.rand(num_images, *image_size).astype(np.uint8) / 255.0
    labels = np.random.randint(0, 10, size=num_images)

    return images.reshape(-1, 28*28), labels


@pytest.fixture
def classifier():
    """Create an instance of the digit classification class."""
    from digits_classifier import ClassifyDigits

    return ClassifyDigits()


def test_digit_recognition_accuracy(model: tf.keras.Model, test_set, classifier):
    """
    Test that the trained model achieves more than 95% accuracy on a given test set.

    Args:
        model (tf.keras.Model): The loaded digit recognition model.
        test_set (tuple[np.ndarray]): A tuple containing images and their corresponding labels.
        classifier: An instance of IClassifyDigits for classifying digits in an image.
    """
    # Extract the images and labels from the test set
    images, expected_labels = test_set

    # Classify the test set using the model
    predicted_labels = np.array([int(np.argmax(model.predict(image.reshape(1, -1)))) for image in images])

    # Calculate accuracy by comparing predicted labels with actual labels
    correct_predictions = sum(predicted_labels == expected_labels)
    accuracy = (correct_predictions / len(expected_labels)) * 100

    assert accuracy > 95


def test_digit_recognition_accuracy_using_classifier(classifier, model: tf.keras.Model):
    """
    Test that the trained model achieves more than 95% accuracy on a given test set using classifier.

    Args:
        classifier: An instance of IClassifyDigits for classifying digits in an image.
        model (tf.keras.Model): The loaded digit recognition model.
    """

    # Generate sample images
    num_images = 1000
    image_size = (28, 28)
    test_set_images = np.random.rand(num_images, *image_size).astype(np.uint8) / 255.0

    predicted_labels = classifier(test_set_images)

    expected_labels = [np.argmax(model.predict(image.reshape(1, -1))) for image in test_set_images]

    correct_predictions = sum(predicted_labels == expected_labels)
    accuracy = (correct_predictions / len(expected_labels)) * 100
    assert accuracy > 95


def test_digit_recognition_accuracy_using_classifier_with_pil_image(classifier):
    """
    Test that the trained model achieves more than 95% accuracy on a given PIL image.

    Args:
        classifier: An instance of IClassifyDigits for classifying digits in an image.
    """

    # Generate sample images
    num_images = 1000

    correct_predictions = 0
    total_labels = 0

    for i in range(num_images):
        pil_image_path = f"image_{i}.png"
        
        if not os.path.exists(pil_image_path):
            continue
        
        image_array = np.array(Image.open(pil_image_path).convert('L').resize((28, 28)))
        predicted_label = classifier(np.expand_dims(image_array, axis=0))[0]

        # Assuming the label is embedded in the filename
        expected_label = int(os.path.splitext(os.path.basename(pil_image_path))[0].split("_")[-1])

        if predicted_label == expected_label:
            correct_predictions += 1

    accuracy = (correct_predictions / num_images) * 100
    assert accuracy > 95


def test_digit_recognition_accuracy_using_classifier_with_pil_multiple_images(classifier):
    """
    Test that the trained model achieves more than 95% accuracy on a given PIL multiple images.

    Args:
        classifier: An instance of IClassifyDigits for classifying digits in an image.
    """

    # Generate sample images
    num_images = 1000

    correct_predictions = 0
    total_labels = 0

    pil_image_paths = [f"image_{i}.png" for i in range(num_images)]

    all_predicted_labels = []
    
    test_set_images = []

    for path in pil_image_paths:
        if not os.path.exists(path):
            continue
        
        image_array = np.array(Image.open(path).convert('L').resize((28, 28)))
        
        # Assuming the label is embedded in the filename
        expected_label = int(os.path.splitext(os.path.basename(path))[0].split("_")[-1])
        
        test_set_images.append(image_array)

    predicted_labels = classifier(np.stack(test_set_images))

    for i, (predicted_label, image_path) in enumerate(zip(predicted_labels, pil_image_paths)):
        if not os.path.exists(image_path):
            continue
        
        # Assuming the label is embedded in the filename
        expected_label = int(os.path.splitext(os.path.basename(image_path))[0].split("_")[-1])

        all_predicted_labels.append((predicted_label, expected_label))

    correct_predictions = sum([label[0] == label[1] for label in all_predicted_labels])
    
    accuracy = (correct_predictions / len(all_predicted_labels)) * 100
    assert accuracy > 95

