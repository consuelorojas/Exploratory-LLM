import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import digits_classifier.constants as constants
import pytest
from sklearn.metrics import accuracy_score


@pytest.fixture
def test_data():
    # Load MNIST dataset for testing (you can use any other method to get your data)
    from tensorflow.keras.datasets import mnist

    (_, _), (x_test, y_test) = mnist.load_data()

    x_test = x_test / 255.0                 # normalize
    x_test = x_test.reshape(-1, 28 * 28)    # flatten

    return x_test, y_test


def test_model_accuracy(test_data):
    model = load_model(constants.MODEL_DIGIT_RECOGNITION_PATH)
    
    images, labels = test_data
    
    predictions = np.argmax(model.predict(images), axis=1)

    accuracy = accuracy_score(labels, predictions)

    assert accuracy >= 0.95
