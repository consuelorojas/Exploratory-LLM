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
    correct_count = 0
    
    for image_path in image_paths:
        x = PIL.Image.open(image_path).convert('L').resize((28, 28))
        images = np.array(x)
        
        prediction = classifier(images=images)
        
        # Assuming the actual digit is in the filename
        expected_digit = int(image_path.split("_")[-1].split(".")[0])
        
        if prediction == expected_digit:
            correct_count += 1
    
    accuracy = (correct_count / len(image_paths)) * 100
    
    assert accuracy > 95


def test_classify_digits_interface() -> None:
    """
    Test the ClassifyDigits class interface.
    
    This ensures that it adheres to the IClassifyDigits interface and can handle
    multiple images at once.
    """

    classifier = ClassifyDigits()
    
    # Create a batch of two images with known digits (e.g., 0 and 1)
    image_0_path = "path_to_image_0.png"
    x_0 = PIL.Image.open(image_0_path).convert('L').resize((28, 28))
    image_1_path = "path_to_image_1.png"
    x_1 = PIL.Image.open(image_1_path).convert('L').resize((28, 28))
    
    images_batch: NDArray[np.uint8] = np.stack([np.array(x_0), np.array(x_1)])
    
    predictions = classifier(images=images_batch)
    
    # Assuming the actual digits are in the filenames
    expected_digits = [
        int(image_path.split("_")[-1].split(".")[0])
        for image_path in [image_0_path, image_1_path]
    ]
    
    assert np.array_equal(predictions, expected_digits)


def test_classify_digits_input_validation() -> None:
    """
    Test the input validation of the ClassifyDigits class.
    
    This ensures that it raises an error when given invalid inputs (e.g., non-image data).
    """

    classifier = ClassifyDigits()
    
    # Try passing a string instead of image data
    with pytest.raises(TypeError):
        classifier(images="invalid_input")
