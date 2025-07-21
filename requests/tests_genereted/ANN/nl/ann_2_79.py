import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))
from main import ClassifyDigits


import numpy as np
from PIL import Image
import tensorflow as tf
import digits_classifier.constants as constants
from digits_classifier.interfaces import IClassifyDigits
from sklearn.metrics import accuracy_score
from unittest.mock import patch, MagicMock
import pytest
from digits_classifier.classifier import ClassifyDigits  # Import the class

# Load MNIST dataset for testing
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

class TestClassifyDigits:
    def test_accuracy(self):
        classify_digits = ClassifyDigits()
        
        # Preprocess images
        x_test_preprocessed = x_test / 255.0
        x_test_preprocessed = x_test_preprocessed.reshape(-1, 28 * 28)
        
        predictions = np.array([int(np.argmax(prediction)) for prediction in 
                                classify_digits.model.predict(x_test_preprocessed)])
        
        accuracy = accuracy_score(y_test, predictions)
        
        assert accuracy >= 0.95

    def test_input_shape(self):
        classify_digits = ClassifyDigits()
        
        # Test with incorrect input shape
        images = np.random.rand(1, 28)  # Incorrect shape
        
        with pytest.raises(ValueError):
            classify_digits(images)

    @patch('digits_classifier.classifier.ClassifyDigits')
    def test_call_method(self, mock_classify_digits):
        mock_instance = MagicMock()
        mock_classify_digits.return_value = mock_instance
        images = np.random.rand(1, 28 * 28)
        
        # Test the call method with correct input shape
        classify_digits = ClassifyDigits()(images)

    def test_model_loading(self):
        model_path = constants.MODEL_DIGIT_RECOGNITION_PATH
        
        try:
            tf.keras.models.load_model(model_path)
        except OSError as e:
            pytest.fail(f"Model loading failed: {e}")

def test_class_instantiation():
    classify_digits = ClassifyDigits()
    
    assert isinstance(classify_digits, IClassifyDigits)

