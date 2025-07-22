import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






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

        assert accuracy >= 95, f"Classification accuracy is {accuracy:.2f}%, which is less than the required 95%"

def get_test_images() -> np.ndarray:
    # Load and preprocess test images
    image_paths = ["path/to/image1.png", "path/to/image2.png"]  # Replace with actual paths to your test images
    
    images = []
    for path in image_paths:
        img = Image.open(path).convert('L').resize((28, 28))
        img_array = np.array(img)
        images.append(img_array)

    return np.array(images)


def get_expected_results() -> np.ndarray[np.int_]:
    # Return expected results (labels) for the test images
    labels = [1, 2]  # Replace with actual labels of your test images
    
    return np.array(labels)
