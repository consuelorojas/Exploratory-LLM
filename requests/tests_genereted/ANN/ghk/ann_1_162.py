import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))


# tests/test_digit_recognition.py

import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
from tensorflow.keras.models import load_model
import PIL.Image


@pytest.fixture
def model():
    """Load the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generate a sample test set for demonstration purposes."""
    # Replace this with your actual test data loading logic.
    images = np.random.rand(100, 28, 28)
    labels = np.random.randint(0, 10, size=100)
    return images, labels


@pytest.fixture
def classifier():
    """Create an instance of the digit classification class."""
    from digits_classifier import ClassifyDigits

    return ClassifyDigits()


def test_digit_recognition_accuracy(model: load_model, test_set, classifier):
    """
    Test that the trained model achieves more than 95% accuracy on a given test set.

    Given:
        - A trained digit recognition model
        - A test set of images and corresponding labels

    When:
        - The test set is classified using the provided model

    Then:
        - An accuracy of more than 95 percent should be achieved.
    """
    # Load the data from the fixture
    images, expected_labels = test_set

    # Normalize and flatten the input as per the classifier's requirements
    normalized_images = (images / 255.0).reshape(-1, 28 * 28)

    # Use the model to make predictions on the flattened images
    predicted_labels = np.array([int(np.argmax(prediction)) for prediction in model.predict(normalized_images)])

    # Alternatively, use the classifier instance directly as per its interface definition.
    # This approach is more aligned with how you might actually be using your ClassifyDigits class,
    # but it requires ensuring that images are properly preprocessed before being passed to the classifier.
    predicted_labels_classifier = classifier(images=np.array([PIL.Image.fromarray(image).convert('L').resize((28, 28)) for image in (images * 255.0).astype(np.uint8)]))

    # Calculate accuracy
    model_accuracy = np.sum(predicted_labels == expected_labels) / len(expected_labels)
    classifier_accuracy = np.sum(predicted_labels_classifier[0] == expected_labels[:len(predicted_labels_classifier)]) / len(expected_labels[:len(predicted_labels_classifier)])

    assert model_accuracy > 0.95, f"Model accuracy {model_accuracy} is less than the required threshold of 95%"
    # Note: The classifier's test might not be directly comparable due to differences in preprocessing steps.
