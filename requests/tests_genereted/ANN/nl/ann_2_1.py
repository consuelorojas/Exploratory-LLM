import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))


import numpy as np
from PIL import Image
import tensorflow as tf
import digits_classifier.constants as constants
from digits_classifier.interfaces import IClassifyDigits
from tests.test_data import get_test_images, get_expected_results


class TestDigitClassification:
    def test_classification_accuracy(self):
        # Load the model and create an instance of ClassifyDigits
        class ClassifyDigits(IClassifyDigits):
            def __call__(self, images: np.ndarray) -> np.ndarray[np.int_]:
                model = tf.keras.models.load_model(constants.MODEL_DIGIT_RECOGNITION_PATH)
                
                images = images / 255.0                 # normalize
                images = images.reshape(-1, 28 * 28)    # flatten

                predictions = model.predict(images)
                return np.array([int(np.argmax(prediction)) for prediction in predictions])

        classify_digits = ClassifyDigits()

        # Get test data and expected results
        test_images = get_test_images()
        expected_results = get_expected_results()

        # Make predictions on the test images
        predicted_results = classify_digits(test_images)

        # Calculate accuracy
        correct_predictions = np.sum(predicted_results == expected_results)
        total_predictions = len(expected_results)
        accuracy = (correct_predictions / total_predictions) * 100

        assert accuracy >= 95, f"Model accuracy is {accuracy:.2f}%, which is less than the required 95%"

def get_test_images() -> np.ndarray:
    # Load test images from a file or database
    # For this example, we'll assume you have a function that loads and preprocesses the images
    return np.array([np.array(Image.open(f"test_image_{i}.png").convert('L').resize((28, 28))) for i in range(100)])

def get_expected_results() -> np.ndarray:
    # Load expected results from a file or database
    # For this example, we'll assume you have a function that loads the expected results
    return np.array([i % 10 for i in range(100)])
