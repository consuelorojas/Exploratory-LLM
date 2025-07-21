import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'ANN'))
from main import ClassifyDigits


import numpy as np
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH, TEST_IMAGES_PATH
import tensorflow as tf
import PIL.Image
import pytest
from sklearn.metrics import accuracy_score


@pytest.fixture
def classify_digits():
    model = tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)
    
    class ClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            images = images / 255.0                 # normalize
            images = images.reshape(-1, 28 * 28)    # flatten

            predictions = model.predict(images)
            return np.array([int(np.argmax(prediction)) for prediction in predictions])

    yield ClassifyDigits()


def test_model_accuracy(classify_digits: IClassifyDigits):
    # Load the test dataset
    (x_test, y_test), _ = tf.keras.datasets.mnist.load_data()
    
    # Preprocess images to match input shape of model
    x_test = np.array([PIL.Image.fromarray(image).resize((28, 28)) for image in x_test])
    x_test = np.array(x_test)[:, :, :, 0]  # Convert to grayscale
    
    predictions = classify_digits(images=x_test)
    
    accuracy = accuracy_score(y_true=y_test, y_pred=predictions)
    
    assert accuracy >= 0.95
