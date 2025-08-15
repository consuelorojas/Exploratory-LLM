import tensorflow as tf

import pytest
from numpy.typing import NDArray
import numpy as np
from PIL.Image import Image
from your_module import ClassifyDigits, constants  # Replace 'your_module' with actual module name


@pytest.fixture
def model() -> tf.keras.Model:
    """Loads the digit recognition model."""
    return tf.keras.models.load_model(constants.MODEL_DIGIT_RECOGNITION_PATH, compile=False)


@pytest.mark.parametrize("digit", range(10))
def test_recognize_single_digit(digit: int) -> None:
    """
    Tests if a single-digit image is recognized correctly.

    Args:
        digit (int): The expected digit.
    """
    # Create an array representing the digit
    img = np.zeros((28, 28), dtype=np.uint8)
    
    # Simulate writing the digit in the center of the image
    for i in range(10):
        for j in range(10):
            if (i - 5) ** 2 + (j - 5) ** 2 < 25:
                img[14 - i, 14 - j] = 255
    
    # Create a ClassifyDigits instance
    classifier = ClassifyDigits()
    
    # Test the classification of the digit image
    assert classifier(np.array([img])) == np.array([digit])


def test_recognize_multiple_digits() -> None:
    """
    Tests if multiple-digit images are recognized correctly.
    """
    # Create an array representing a sequence of digits (e.g., 0-9)
    img_sequence = []
    
    for digit in range(10):
        img = np.zeros((28, 28), dtype=np.uint8)
        
        # Simulate writing the digit in the center of the image
        for i in range(10):
            for j in range(10):
                if (i - 5) ** 2 + (j - 5) ** 2 < 25:
                    img[14 - i, 14 - j] = 255
        
        # Append each digit to the sequence
        img_sequence.append(img)
    
    classifier = ClassifyDigits()
    
    predictions = classifier(np.array(img_sequence))
    
    assert np.all(predictions == range(10))


def test_recognition_accuracy() -> None:
    """
    Tests if at least 95% of input digits are recognized correctly.
    """
    # Generate a dataset with various single-digit numbers (0 through 9)
    num_digits = 100
    img_dataset: list[NDArray] = []
    
    for _ in range(num_digits):
        digit_img = np.zeros((28, 28), dtype=np.uint8)
        
        # Simulate writing the digit in the center of the image
        for i in range(10):
            for j in range(10):
                if (i - 5) ** 2 + (j - 5) ** 2 < 25:
                    digit_img[14 - i, 14 - j] = np.random.randint(0, 256)
        
        img_dataset.append(digit_img)
    
    # Create a ClassifyDigits instance
    classifier = ClassifyDigits()
    
    predictions: NDArray[np.int_] = classifier(np.array(img_dataset))
    
    correct_count = sum(predictions == [np.argmax(np.random.rand(10)) for _ in range(num_digits)])
    
    accuracy = (correct_count / num_digits) * 100
    
    assert accuracy >= 95
