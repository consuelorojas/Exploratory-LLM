
import pytest
from numpy.typing import NDArray
import tensorflow as tf
import interfaces
import PIL.Image
import numpy as np
import constants
from your_module import ClassifyDigits  # Replace 'your_module' with actual module name

# Load the model only once for all tests
model_path = constants.MODEL_DIGIT_RECOGNITION_PATH
model = tf.keras.models.load_model(model_path, compile=False)

class TestDigitRecognition:
    @pytest.fixture
    def classify_digits(self):
        return ClassifyDigits()

    # Create a test dataset with at least 10 different inputs (0-9)
    @pytest.mark.parametrize("image_path, expected_digit", [
        ("path_to_image_0.png", 0),
        ("path_to_image_1.png", 1),
        ("path_to_image_2.png", 2),
        ("path_to_image_3.png", 3),
        ("path_to_image_4.png", 4),
        ("path_to_image_5.png", 5),
        ("path_to_image_6.png", 6),
        ("path_to_image_7.png", 7),
        ("path_to_image_8.png", 8),
        ("path_to_image_9.png", 9)
    ])
    def test_recognize_digits(self, classify_digits: ClassifyDigits, image_path: str, expected_digit: int):
        # Load the image and convert it to a numpy array
        x = PIL.Image.open(image_path).convert('L').resize((28, 28))
        images = np.array(x)

        # Make predictions using the model
        predicted_digits = classify_digits(images=images)

        assert len(predicted_digits) == 1
        assert predicted_digits[0] == expected_digit

    def test_recognition_accuracy(self, classify_digits: ClassifyDigits):
        # Create a dataset with at least 100 different inputs (0-9)
        image_paths = [f"path_to_image_{i}.png" for i in range(10)] * 10
        expected_digits = list(range(10)) * 10

        correct_predictions = 0
        total_images = len(image_paths)

        for image_path, expected_digit in zip(image_paths, expected_digits):
            x = PIL.Image.open(image_path).convert('L').resize((28, 28))
            images = np.array(x)
            predicted_digits = classify_digits(images=images)

            if predicted_digits[0] == expected_digit:
                correct_predictions += 1

        accuracy = (correct_predictions / total_images) * 100
        assert accuracy > 95.0
