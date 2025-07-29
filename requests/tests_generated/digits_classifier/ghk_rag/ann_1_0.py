
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
    @pytest.fixture
    def test_dataset(self):
        images = []
        labels = []

        for i in range(10):
            img_path = f"test_image_{i}.png"
            x = PIL.Image.open(img_path).convert('L').resize((28, 28))
            image_array = np.array(x)
            images.append(image_array)
            labels.append(i)

        return images, labels

    def test_recognition_accuracy(self, classify_digits: ClassifyDigits, test_dataset):
        # Unpack the dataset
        images, labels = test_dataset

        correct_predictions = 0
        total_images = len(images)

        for image, label in zip(images, labels):
            prediction = classify_digits(np.array([image]))
            if int(prediction[0]) == label:
                correct_predictions += 1

        accuracy = (correct_predictions / total_images) * 100
        assert accuracy > 95.0

    def test_invalid_input_type(self, classify_digits: ClassifyDigits):
        with pytest.raises(TypeError):
            # Pass a string instead of an array to simulate invalid input type
            classify_digits("invalid_input")

    @pytest.mark.parametrize(
        "image_array",
        [
            np.random.rand(1, 28 * 28),  # Random pixel values (not necessarily digits)
            np.zeros((1, 28 * 28)),     # All black pixels
            np.ones((1, 28 * 28))       # All white pixels
        ]
    )
    def test_edge_cases(self, classify_digits: ClassifyDigits, image_array):
        prediction = classify_digits(image_array)
        assert isinstance(prediction[0], int) and 0 <= prediction[0] < 10

# Example usage of the above tests:
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
