import tensorflow as tf

import pytest
from numpy.typing import NDArray
import numpy as np
from PIL.Image import Image
from your_module import ClassifyDigits, constants  # Replace 'your_module' with actual module name


@pytest.fixture
def model():
    return tf.keras.models.load_model(constants.MODEL_DIGIT_RECOGNITION_PATH, compile=False)


@pytest.mark.parametrize("image_path", [
    "path_to_image_0.png",
    "path_to_image_1.png",
    "path_to_image_2.png",
    # Add more image paths here
])
def test_recognize_digits(image_path: str) -> None:
    """
    Test the digit recognition function with a single input.
    
    Args:
        image_path (str): Path to an image file containing a single-digit number.
    """

    x = PIL.Image.open(image_path).convert('L').resize((28, 28))
    images = np.array(x)

    classifier = ClassifyDigits()
    prediction = classifier(images=images)
    
    # Assuming the actual digit is in the filename
    expected_digit = int(image_path.split("_")[-1].split(".")[0])
    
    assert prediction == expected_digit


def test_recognize_digits_bulk() -> None:
    """
    Test the digit recognition function with multiple inputs.
    """

    image_paths = [
        "path_to_image_0.png",
        "path_to_image_1.png",
        # Add more image paths here
    ]

    classifier = ClassifyDigits()
    
    correct_predictions = 0
    
    for image_path in image_paths:
        x = PIL.Image.open(image_path).convert('L').resize((28, 28))
        images = np.array(x)
        
        prediction = classifier(images=images)
        
        # Assuming the actual digit is in the filename
        expected_digit = int(image_path.split("_")[-1].split(".")[0])
        
        if prediction == expected_digit:
            correct_predictions += 1
    
    accuracy = (correct_predictions / len(image_paths)) * 100
    
    assert accuracy > 95


def test_recognize_digits_invalid_input() -> None:
    """
    Test the digit recognition function with invalid input.
    """

    classifier = ClassifyDigits()
    
    # Invalid image shape
    images = np.random.rand(1, 10)
    
    with pytest.raises(ValueError):
        classifier(images=images)


def test_recognize_digits_empty_input() -> None:
    """
    Test the digit recognition function with empty input.
    """

    classifier = ClassifyDigits()
    
    # Empty image array
    images: NDArray[np.int_] = np.array([])
    
    with pytest.raises(ValueError):
        classifier(images=images)
