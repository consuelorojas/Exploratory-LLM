import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))


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
    images = np.random.rand(10, 784).astype(np.float32) / 255.0
    predictions = model.predict(images.reshape(-1, 28, 28))
    
    assert len(predictions.shape) == 2 and predictions.shape[1] == 10


def test_model_input_shape(model):
    images = np.random.rand(10, 784).astype(np.float32)
    with pytest.raises(ValueError):
        model.predict(images)

    # Reshape to correct input shape
    reshaped_images = images.reshape(-1, 28 * 28) / 255.0
    
    try:
        predictions = model.predict(reshaped_images.reshape(-1, 28, 28))
    except ValueError as e:
        pytest.fail(f"Model failed with error: {e}")
