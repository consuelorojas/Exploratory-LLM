
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

class TestClassifyDigits:
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
        
        assert len(predicted_digits) == 1  # Only one image is passed to the function

    def test_recognition_accuracy(self, classify_digits: ClassifyDigits):
        correct_count = 0
        total_images = 10
        
        for i in range(total_images):
            image_path = f"path_to_image_{i}.png"
            expected_digit = i
            
            # Load the image and convert it to a numpy array
            x = PIL.Image.open(image_path).convert('L').resize((28, 28))
            images = np.array(x)

            # Make predictions using the model
            predicted_digits = classify_digits(images=images)
            
            if len(predicted_digits) > 0:
                assert isinstance(predicted_digits[0], int), "Predicted digit should be an integer"
                
                correct_count += (predicted_digits[0] == expected_digit)

        accuracy = correct_count / total_images
        assert accuracy >= 0.95, f"Recognition accuracy is {accuracy:.2f}, which is less than the required 95%"

    def test_invalid_input(self, classify_digits: ClassifyDigits):
        # Test with an empty numpy array
        images = np.array([])
        
        with pytest.raises(ValueError) as exc_info:
            classify_digits(images=images)
            
        assert str(exc_info.value).startswith("Input image is invalid")

# Note that you need to replace "path_to_image_X.png" and 'your_module' in the code above 
