import tensorflow as tf
import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






import numpy as np
from tensorflow.keras.models import load_model
import digits_classifier.constants as constants
import pytest

# Load test dataset (e.g., MNIST)
def get_test_dataset():
    # Replace with your actual method to load and preprocess the test data
    from tensorflow.keras.datasets import mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    
    x_test = x_test / 255.0                 # normalize
    x_test = x_test.reshape(-1, 28 * 28)    # flatten
    
    return x_test, y_test

# Load the model and test its accuracy on a given dataset
def get_model_accuracy(model_path: str):
    model = load_model(model_path)
    
    x_test, y_test = get_test_dataset()
    
    predictions = np.argmax(model.predict(x_test), axis=1)
    correct_predictions = sum(predictions == y_test)
    
    return (correct_predictions / len(y_test)) * 100

# Pytest to check the model accuracy
def test_model_accuracy():
    min_required_accuracy = 95.0
    
    # Load and evaluate the model's performance on a given dataset
    actual_accuracy = get_model_accuracy(constants.MODEL_DIGIT_RECOGNITION_PATH)
    
    assert actual_accuracy >= min_required_accuracy, (
        f"Model accuracy ({actual_accuracy:.2f}%) is less than required {min_required_accuracy}%"
    )
