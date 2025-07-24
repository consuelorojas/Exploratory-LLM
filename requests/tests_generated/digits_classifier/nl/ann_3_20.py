import tensorflow as tf
import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






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
    # Generate random MNIST-like data for testing (replace with actual test dataset)
    np.random.seed(0)
    images = np.random.rand(test_size, 28 * 28).astype(np.float32) / 255.0
    labels = np.random.randint(10, size=test_size)

    predictions = model.predict(images.reshape(-1, 28, 28))
    predicted_labels = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predicted_labels)
    assert accuracy >= 0.95


def test_model_output_shape(model):
    images = np.random.rand(10, 28 * 28).astype(np.float32) / 255.0
    predictions = model.predict(images.reshape(-1, 28, 28))
    
    assert len(predictions.shape) == 2 and predictions.shape[1] == 10


def test_model_input_shape(model):
    images = np.random.rand(10, 784).astype(np.float32)
    with pytest.raises(ValueError):
        model.predict(images)

    images = np.random.rand(10, 28 * 28).reshape(-1, 28, 28).astype(np.float32) / 255.0
    predictions = model.predict(images)
    
    assert len(predictions.shape) == 2 and predictions.shape[1] == 10


def test_model_output_type(model):
    images = np.random.rand(10, 784).reshape(-1, 28 * 28).astype(np.float32) / 255.0
    predictions = model.predict(images)
    
    assert isinstance(predictions, np.ndarray)


@pytest.mark.parametrize("input_shape", [(100,), (50, 20), (5,)])
def test_model_input_type(model, input_shape):
    images = np.random.rand(*input_shape).astype(np.float32) / 255.0
    with pytest.raises(ValueError):
        model.predict(images.reshape(-1))
