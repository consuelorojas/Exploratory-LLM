
import pytest
from numpy.typing import NDArray
import tensorflow as tf
import interfaces
import PIL.Image
import numpy as np
import constants
from your_module import ClassifyDigits  # Replace 'your_module' with actual module name


def test_recognize_digits():
    """
    Test the digit recognition function.
    
    This test checks if the function can recognize more than 95% of digits correctly.
    It uses a dataset containing various single-digit numbers (0 through 9) and other random string data.
    The test is run with at least ten different inputs from the dataset.
    """
    # Load model
    print("Loading model from:", constants.MODEL_DIGIT_RECOGNITION_PATH)
    model_path = constants.MODEL_DIGIT_RECOGNITION_PATH
    model = tf.keras.models.load_model(model_path, compile=False)

    # Create an instance of ClassifyDigits class
    classify_digits = ClassifyDigits()

    # Define test dataset (replace with actual paths to images or use a library like MNIST)
    image_paths = [
        "path_to_image_0.png",
        "path_to_image_1.png",
        "path_to_image_2.png",
        "path_to_image_3.png",
        "path_to_image_4.png",
        "path_to_image_5.png",
        "path_to_image_6.png",
        "path_to_image_7.png",
        "path_to_image_8.png",
        "path_to_image_9.png"
    ]

    # Define expected outputs
    expected_outputs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    correct_recognitions = 0

    for image_path, expected_output in zip(image_paths, expected_outputs):
        x = PIL.Image.open(image_path).convert('L').resize((28, 28))
        images = np.array(x)
        
        # Get the predicted output
        prediction = classify_digits(images=images)[0]

        if prediction == expected_output:
            correct_recognitions += 1

    recognition_accuracy = (correct_recognitions / len(image_paths)) * 100

    assert recognition_accuracy > 95


def test_classify_digits_interface():
    """
    Test the ClassifyDigits class interface.
    
    This test checks if the ClassifyDigits class has a __call__ method and 
    it returns an NDArray of integers as expected.
    """

    classify_digits = ClassifyDigits()

    # Create dummy input
    images: NDArray = np.random.rand(1, 28 * 28)

    output = classify_digits(images=images)

    assert isinstance(output, NDArray)
    assert output.dtype == np.int_


def test_classify_digits_input_validation():
    """
    Test the ClassifyDigits class input validation.
    
    This test checks if the ClassifyDigits class raises an error when 
    invalid inputs are provided (e.g., non-NDArray or incorrect shape).
    """

    classify_digits = ClassifyDigits()

    # Create dummy input with wrong type
    images: str = "invalid_input"

    with pytest.raises(TypeError):
        classify_digits(images=images)

    # Create dummy input with correct type but wrong shape
    images: NDArray = np.random.rand(1, 10 * 20)  # Wrong size

    with pytest.warns(UserWarning):  # Assuming a warning is raised for incorrect shape
        classify_digits(images=images)
