import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))


import numpy as np
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH, TEST_IMAGES_PATH
import tensorflow as tf
import PIL.Image
import pytest
from sklearn.metrics import accuracy_score


@pytest.fixture
def classify_digits() -> IClassifyDigits:
    model = tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)
    
    class ClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray[np.int_]:
            images = images / 255.0                 # normalize
            images = images.reshape(-1, 28 * 28)    # flatten

            predictions = model.predict(images)
            return np.array([int(np.argmax(prediction)) for prediction in predictions])
    
    yield ClassifyDigits()


def test_classification_accuracy(classify_digits: IClassifyDigits):
    # Load the test dataset
    (x_test, y_test) = tf.keras.datasets.mnist.load_data()[1]
    x_test = x_test.reshape(-1, 28, 28)
    
    predictions = classify_digits(np.array([PIL.Image.fromarray(image).convert('L').resize((28, 28)) for image in x_test]))
    
    accuracy = accuracy_score(y_test[:len(predictions)], predictions)
    assert round(accuracy * 100) >= 95
