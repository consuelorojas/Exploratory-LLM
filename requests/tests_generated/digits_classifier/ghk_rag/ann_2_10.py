
import pytest
from numpy.typing import NDArray
import tensorflow as tf
import interfaces
import PIL.Image
import numpy as np
import constants
from your_module import ClassifyDigits  # Replace 'your_module' with actual module name


# Load the model only once for all tests
@pytest.fixture(scope="session")
def loaded_model():
    print("Loading model from:", constants.MODEL_DIGIT_RECOGNITION_PATH)
    return tf.keras.models.load_model(constants.MODEL_DIGIT_RECOGNITION_PATH, compile=False)


class TestDigitRecognition:
    @pytest.mark.parametrize(
        "image_path, expected_digit",
        [
            ("path_to_image_0.png", 0),
            ("path_to_image_1.png", 1),
            # Add more test cases here...
            ("path_to_image_9.png", 9),
        ],
    )
    def test_recognize_digits(self, loaded_model: tf.keras.Model, image_path: str, expected_digit: int):
        x = PIL.Image.open(image_path).convert('L').resize((28, 28))
        images = np.array(x)
        classifier = ClassifyDigits()
        recognized_digit = classifier(images=images)[0]
        assert recognized_digit == expected_digit

    def test_recognize_multiple_digits(self, loaded_model: tf.keras.Model):
        # Create a list of image paths and corresponding digits
        image_paths_and_expected_digits = [
            ("path_to_image_0.png", 0),
            ("path_to_image_1.png", 1),
            # Add more test cases here...
            ("path_to_image_9.png", 9),
        ]

        correct_recognitions = 0

        for image_path, expected_digit in image_paths_and_expected_digits:
            x = PIL.Image.open(image_path).convert('L').resize((28, 28))
            images = np.array(x)
            classifier = ClassifyDigits()
            recognized_digit = classifier(images=images)[0]
            if recognized_digit == expected_digit:
                correct_recognitions += 1

        recognition_accuracy = (correct_recognitions / len(image_paths_and_expected_digits)) * 100
        assert recognition_accuracy > 95


def test_classify_digits_interface():
    # Test that the ClassifyDigits class implements the IClassifyDigits interface
    classifier = ClassifyDigits()
    image_path = "path_to_image_0.png"
    x = PIL.Image.open(image_path).convert('L').resize((28, 28))
    images = np.array(x)
    assert isinstance(classifier(images=images), NDArray)


def test_classify_digits_input_validation():
    # Test that the ClassifyDigits class raises an error for invalid input
    classifier = ClassifyDigits()
    with pytest.raises(ValueError):
        classifier(images=None)  # Replace 'None' with any other invalid input type


# Example usage of a parametrized test to check recognition accuracy on multiple images:
@pytest.mark.parametrize(
    "image_path, expected_digit",
    [
        ("path_to_image_0.png", 0),
        ("path_to_image_1.png", 1),
        # Add more test cases here...
        ("path_to_image_9.png", 9),
    ],
)
def test_recognition_accuracy(image_path: str, expected_digit: int):
    x = PIL.Image.open(image_path).convert('L').resize((28, 28))
    images = np.array(x)
    classifier = ClassifyDigits()
    recognized_digit = classifier(images=images)[0]
    assert recognized_digit == expected_digit
