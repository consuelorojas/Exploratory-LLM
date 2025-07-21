import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))
from main import ClassifyDigits


import numpy as np
from digits_classifier.interfaces import IClassifyDigits
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from PIL.Image import Image
import pytest

class TestDigitClassification:
    @pytest.fixture
    def classifier(self):
        return ClassifyDigits()

    @pytest.fixture
    def test_data(self):
        (x_train, y_train), (_, _) = mnist.load_data()
        # Select a subset of the data for testing
        indices = np.random.choice(x_train.shape[0], size=100)
        x_test = x_train[indices]
        y_test = y_train[indices]

        return x_test, y_test

    def test_classification_accuracy(self, classifier: IClassifyDigits, test_data):
        images, labels = test_data
        # Preprocess the data to match what's expected by ClassifyDigits()
        preprocessed_images = np.array([np.array(Image.fromarray(image).resize((28, 28))) for image in images])

        predictions = classifier(preprocessed_images)
        accuracy = accuracy_score(labels, predictions)

        assert accuracy >= 0.95
