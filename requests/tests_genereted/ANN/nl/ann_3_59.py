import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))


import numpy as np
from PIL import Image
import tensorflow as tf
import digits_classifier.constants as constants
from digits_classifier.interfaces import IClassifyDigits
from sklearn.metrics import accuracy_score
from unittest.mock import MagicMock, patch
from pytest import fixture
import os


class TestDigitClassification:
    @fixture
    def model(self):
        return tf.keras.models.load_model(constants.MODEL_DIGIT_RECOGNITION_PATH)

    @fixture
    def classify_digits(self) -> IClassifyDigits:
        class ClassifyDigits(IClassifyDigits):
            def __call__(self, images: np.ndarray) -> np.ndarray[np.int_]:
                # normalize and flatten the input image data
                normalized_images = images / 255.0
                flattened_images = normalized_images.reshape(-1, 28 * 28)

                predictions = self.model.predict(flattened_images)
                return np.array([int(np.argmax(prediction)) for prediction in predictions])

            def __init__(self):
                super().__init__()
                self.model = tf.keras.models.load_model(constants.MODEL_DIGIT_RECOGNITION_PATH)

        return ClassifyDigits()

    @fixture
    def test_data(self) -> tuple[np.ndarray, np.ndarray]:
        # Load MNIST dataset for testing (assuming it's downloaded)
        fashion_mnist = tf.keras.datasets.mnist

        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        return test_images.reshape(-1, 28 * 28).reshape(1000, 28, 28), test_labels[:1000]

    def test_accuracy(self, classify_digits: IClassifyDigits, model, test_data):
        # Load MNIST dataset for testing
        images, labels = test_data

        predictions = classify_digits(images)

        accuracy = accuracy_score(labels, predictions)
        
        assert round(accuracy * 100) >= 95


def test_classify_digit():
    image_path = os.path.join(os.getcwd(), "test_image.png")
    
    # Create a sample MNIST-like image
    img_array = np.random.randint(low=0, high=256, size=(28, 28), dtype=np.uint8)
    Image.fromarray(img_array).save(image_path)

    classify_digits = ClassifyDigits()
    result = classify_digits(images=np.array(PIL.Image.open(image_path).convert('L').resize((28, 28))))

    assert isinstance(result[0], int)


class TestClassifyDigit:
    @patch("digits_classifier.interfaces.IClassifyDigits")
    def test_call(self, mock_class):
        # Arrange
        images = np.random.rand(1, 784)
        
        classify_digits_instance: IClassifyDigits = ClassifyDigits()
        
        expected_result = [np.argmax(np.array([0.9]))]

        with patch.object(classify_digits_instance.model, 'predict', return_value=np.array([[0.9]])):
            # Act
            result = classify_digits_instance(images)

            assert np.allclose(result, expected_result)
