import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))
from main import ClassifyDigits


import numpy as np
from tensorflow.keras.models import load_model
import digits_classifier.constants as constants
import PIL.Image
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.classifier import ClassifyDigits
import pytest
import os


@pytest.fixture
def classifier():
    return ClassifyDigits()


@pytest.mark.parametrize("image_path, expected_digit", [
    (os.path.join(constants.TEST_IMAGES_PATH, "0.png"), 0),
    (os.path.join(constants.TEST_IMAGES_PATH, "1.png"), 1),
    # Add more test images here...
])
def test_classification_accuracy(classifier: IClassifyDigits, image_path: str, expected_digit: int):
    prediction = classifier(np.array(PIL.Image.open(image_path).convert('L').resize((28, 28))))
    assert len(prediction) == 1
    assert prediction[0] == expected_digit


def test_model_accuracy(classifier: IClassifyDigits):
    # Load the MNIST dataset for testing (assuming it's stored in a file named 'mnist_test.npy')
    test_images = np.load(constants.TEST_IMAGES_PATH)
    
    predictions = classifier(test_images)

    accuracy = sum(1 for i, prediction in enumerate(predictions) if int(prediction) == constants.get_expected_digits()[i]) / len(predictions)
    assert round(accuracy * 100, 2) >= 95
