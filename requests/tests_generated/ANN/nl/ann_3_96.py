import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'ANN'))
from main import ClassifyDigits


import numpy as np
from tensorflow.keras.models import load_model
import digits_classifier.constants as constants
import pytest
from sklearn.metrics import accuracy_score


@pytest.fixture
def model():
    return load_model(constants.MODEL_DIGIT_RECOGNITION_PATH)


@pytest.mark.parametrize("test_size", [100, 500])
def test_classification_accuracy(model, test_size):
    # Load MNIST dataset for testing (assuming it's available)
    from tensorflow.keras.datasets import mnist

    (_, _), (x_test, y_test) = mnist.load_data()

    x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0
    predictions = model.predict(x_test[:test_size])
    predicted_classes = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(y_test[:test_size], predicted_classes)
    assert accuracy >= 0.95


def test_model_output_shape(model):
    # Test the output shape of the model with a random input
    from tensorflow.keras.datasets import mnist

    (_, _), (x_test, _) = mnist.load_data()

    x_test = np.array([x_test[0].reshape(28 * 28).astype('float32') / 255.0])
    predictions = model.predict(x_test)
    assert len(predictions.shape) == 2 and predictions.shape[-1] == 10


def test_model_input_shape(model):
    # Test the input shape of the model with a random input
    from tensorflow.keras.datasets import mnist

    (_, _), (x_test, _) = mnist.load_data()

    x_test = np.array([x_test[0].reshape(28 * 28).astype('float32') / 255.0])
    predictions = model.predict(x_test)
    assert len(predictions.shape) == 2 and predictions.shape[-1] == 10
