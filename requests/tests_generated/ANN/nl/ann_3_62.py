import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'ANN'))
from main import ClassifyDigits


import numpy as np
from tensorflow.keras.models import load_model
import digits_classifier.constants as constants
import pytest

# Load test dataset (e.g., MNIST)
def get_test_data():
    # Replace with your actual method to load and preprocess test data
    from tensorflow.keras.datasets import mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    
    x_test = x_test / 255.0                 # normalize
    x_test = x_test.reshape(-1, 28 * 28)    # flatten
    
    return x_test, y_test

# Load model and test classification accuracy
def get_model_accuracy(model_path):
    model = load_model(model_path)
    
    x_test, y_test = get_test_data()
    
    predictions = np.argmax(model.predict(x_test), axis=1)
    correct_predictions = sum(predictions == y_test)
    
    return (correct_predictions / len(y_test)) * 100

# Pytest to check classification accuracy
def test_classification_accuracy():
    model_path = constants.MODEL_DIGIT_RECOGNITION_PATH
    
    # Load and calculate the accuracy of the loaded model
    accuracy = get_model_accuracy(model_path)
    
    assert accuracy >= 95, f"Model's accuracy {accuracy} is less than expected (>= 95%)"

# Run pytest to check classification accuracy
pytest.main([__file__, "-v", "--capture=no"])
