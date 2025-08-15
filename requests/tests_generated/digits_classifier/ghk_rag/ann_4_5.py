
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

    # Create a test dataset with images of digits from 0 to 9
    @pytest.fixture
    def digit_images(self):
        import os
        image_paths = []
        for i in range(10):
            img_path = f"test_data/{i}.png"
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"No test data found at {img_path}")
            image_paths.append(PIL.Image.open(img_path).convert('L').resize((28, 28)))
        return [np.array(x) for x in image_paths]

    def test_recognition_accuracy(self, classify_digits: ClassifyDigits, digit_images):
        # Test the recognition accuracy with at least ten different inputs
        predictions = []
        correct_labels = list(range(10)) * (len(digit_images) // 10 + 1)
        for i in range(len(correct_labels)):
            image = np.array([digit_images[i % len(digit_images)]])
            prediction = classify_digits(image)[0]
            predictions.append(prediction)

        # Calculate the accuracy
        correct_predictions = sum(1 for pred, label in zip(predictions[:10], correct_labels[:10]) if pred == label)
        accuracy = (correct_predictions / 10) * 100

        assert accuracy > 95, f"Recognition accuracy is {accuracy:.2f}%, which is less than the required 95%"

    def test_recognition_with_random_data(self, classify_digits: ClassifyDigits):
        # Test with random data to ensure it doesn't crash
        import numpy as np
        random_image = np.random.randint(0, 256, size=(28, 28), dtype=np.uint8)
        image_array = np.array([random_image])
        prediction = classify_digits(image_array)[0]
        assert isinstance(prediction, int) and 0 <= prediction < 10

    def test_recognition_with_empty_input(self, classify_digits: ClassifyDigits):
        # Test with an empty input to ensure it raises the correct error
        image_array = np.array([])
        with pytest.raises(ValueError):
            classify_digits(image_array)

# Example usage:
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
