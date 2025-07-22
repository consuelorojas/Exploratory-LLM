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
        # Load the model and create an instance of ClassifyDigits class
        from digits_classifier.classifier import ClassifyDigits

        classifier = ClassifyDigits()

        # Get test images and expected results
        test_images, expected_results = get_test_images(), get_expected_results()

        # Convert test images to numpy array
        test_images_array = np.array([np.array(Image.open(image_path).convert('L').resize((28, 28))) for image_path in test_images])

        # Make predictions using the classifier
        predicted_results = classifier(test_images_array)

        # Calculate accuracy
        correct_predictions = sum(1 for prediction, expected_result in zip(predicted_results, expected_results) if prediction == expected_result)
        accuracy = (correct_predictions / len(expected_results)) * 100

        assert accuracy >= 95


def get_test_images():
    return [
        'path_to_image_0.png',
        'path_to_image_1.png',
        # Add more image paths here
    ]


def get_expected_results():
    return [5, 3, 
            # Add expected results for each test image
           ]
