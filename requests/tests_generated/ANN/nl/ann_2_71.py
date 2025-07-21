import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'ANN'))
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
        # Create an instance of the ClassifyDigits class
        classify_digits = interfaces.ClassifyDigits()

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        x_test_image = np.array([PIL.Image.fromarray(x_test[0]).convert('L').resize((28, 28))])
        
        prediction = classify_digits(images=x_test_image)
        
        assert isinstance(prediction, np.ndarray)

    def test_classify_digits_output_type(self):
        # Create an instance of the ClassifyDigits class
        classify_digits = interfaces.ClassifyDigits()

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        x_test_image = np.array([PIL.Image.fromarray(x_test[0]).convert('L').resize((28, 28))])
        
        prediction = classify_digits(images=x_test_image)
        
        assert isinstance(prediction.dtype.type(np.int_()), type(int))
