import tensorflow as tf

# tests/test_digit_recognition.py
import pytest
from PIL import Image
import numpy as np
from your_module import ClassifyDigits, constants  # Replace 'your_module' with actual module name

def test_load_model():
    """Test if model is loaded successfully."""
    assert isinstance(constants.MODEL_DIGIT_RECOGNITION_PATH, str)
    try:
        tf.keras.models.load_model(constants.MODEL_DIGIT_RECOGNITION_PATH, compile=False)
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")

@pytest.fixture
def digit_recognizer():
    """Create a ClassifyDigits instance for testing."""
    return ClassifyDigits()

# Assuming you have a dataset of images with their corresponding labels (0-9)
test_images = [
    ("image_1.png", 5),
    ("image_2.png", 3),
    ("image_3.png", 8),
    # Add more test cases here...
]

def test_recognize_digits(digit_recognizer):
    """Test if digits are recognized correctly."""
    correct_count = 0
    for image_path, expected_digit in test_images:
        img = Image.open(image_path).convert('L').resize((28, 28))
        images = np.array(img)
        predicted_digit = digit_recognizer(images=images)[0]
        if predicted_digit == expected_digit:
            correct_count += 1
    accuracy = (correct_count / len(test_images)) * 100
    assert accuracy > 95

def test_invalid_input(digit_recognizer):
    """Test with invalid input."""
    try:
        digit_recognizer(images=np.array([[[[0, 255]]]]))
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")

# Add more tests for edge cases and other scenarios...
