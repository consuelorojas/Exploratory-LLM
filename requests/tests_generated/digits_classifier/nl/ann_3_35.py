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
    # Generate random MNIST-like data for testing purposes.
    np.random.seed(0)
    images = np.random.rand(test_size, 28 * 28).astype(np.float32) / 255.0
    labels = np.random.randint(low=0, high=10, size=test_size)

    predictions = model.predict(images.reshape(-1, 28, 28))
    predicted_labels = np.argmax(predictions, axis=-1)
    
    accuracy = accuracy_score(labels, predicted_labels)
    assert accuracy >= 0.95
