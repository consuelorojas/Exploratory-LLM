import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))
from main import ClassifyDigits


import numpy as np
from PIL import Image
import tensorflow as tf
import digits_classifier.constants as constants
import digits_classifier.interfaces as interfaces
from sklearn.metrics import accuracy_score
import pytest

class TestClassifyDigits:
    def test_model_accuracy(self):
        # Load the model and MNIST dataset for testing
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Normalize pixel values to be between 0 and 1
        x_test = x_test / 255.0

        # Flatten images into a single array of pixels per image
        x_test = x_test.reshape(-1, 28 * 28)

        model = tf.keras.models.load_model(constants.MODEL_DIGIT_RECOGNITION_PATH)
        
        predictions = np.argmax(model.predict(x_test), axis=1)
        
        accuracy = accuracy_score(y_test, predictions)
        
        assert round(accuracy, 2) >= 0.95

    def test_classify_digits_interface(self):
        class ClassifyDigits(interfaces.IClassifyDigits):
            def __call__(self, images: np.ndarray) -> np.ndarray:
                model = tf.keras.models.load_model(constants.MODEL_DIGIT_RECOGNITION_PATH)
                
                # Normalize pixel values to be between 0 and 1
                images = images / 255.0
                
                # Flatten images into a single array of pixels per image
                images = images.reshape(-1, 28 * 28)

                predictions = model.predict(images)
                return np.array([int(np.argmax(prediction)) for prediction in predictions])

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        classify_digits = ClassifyDigits()

        # Test the interface with a single image
        test_image = x_test[0]
        predicted_digit = classify_digits(np.array([test_image]))
        
        assert isinstance(predicted_digit, np.ndarray)
        assert len(predicted_digit.shape) == 1

        # Test the accuracy of the classification using multiple images
        predictions = classify_digits(x_test[:100])
        accuracy = sum(1 for i in range(len(predictions)) if y_test[i] == predictions[i]) / len(y_test[:100])

        assert round(accuracy, 2) >= 0.95

def test_invalid_input():
    class ClassifyDigits(interfaces.IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            model = tf.keras.models.load_model(constants.MODEL_DIGIT_RECOGNITION_PATH)
            
            # Normalize pixel values to be between 0 and 1
            images = images / 255.0
            
            # Flatten images into a single array of pixels per image
            images = images.reshape(-1, 28 * 28)

            predictions = model.predict(images)
            return np.array([int(np.argmax(prediction)) for prediction in predictions])

    classify_digits = ClassifyDigits()

    with pytest.raises(ValueError):
        classify_digits(np.random.rand(10))

def test_empty_input():
    class ClassifyDigits(interfaces.IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            model = tf.keras.models.load_model(constants.MODEL_DIGIT_RECOGNITION_PATH)
            
            # Normalize pixel values to be between 0 and 1
            images = images / 255.0
            
            # Flatten images into a single array of pixels per image
            images = images.reshape(-1, 28 * 28)

            predictions = model.predict(images)
            return np.array([int(np.argmax(prediction)) for prediction in predictions])

    classify_digits = ClassifyDigits()

    with pytest.raises(ValueError):
        classify_digits(np.empty((0)))
