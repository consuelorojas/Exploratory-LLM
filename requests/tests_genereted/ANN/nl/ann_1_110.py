import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))


import numpy as np
from digits_classifier.interfaces import IClassifyDigits
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from PIL.Image import Image

class TestDigitClassification:
    def test_classification_accuracy(self):
        # Load MNIST dataset for testing
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Resize and normalize images
        x_test_resized = np.array([np.array(Image.fromarray(image).resize((28, 28))) for image in x_test])
        x_test_normalized = x_test_resized / 255.0

        # Create an instance of the ClassifyDigits class
        classifier: IClassifyDigits = ClassifyDigits()

        # Make predictions on test data
        y_pred = classifier(x_test_normalized)

        # Calculate accuracy using sklearn's accuracy_score function
        accuracy = accuracy_score(y_test, y_pred)

        # Assert that the classification accuracy is at least 95%
        assert accuracy >= 0.95

    def test_classification_output_type(self):
        (x_train, _), (_, _) = mnist.load_data()
        
        x_test_resized = np.array([np.array(Image.fromarray(image).resize((28, 28))) for image in [x_train[0]]])
        x_test_normalized = x_test_resized / 255.0

        classifier: IClassifyDigits = ClassifyDigits()

        y_pred = classifier(x_test_normalized)

        assert isinstance(y_pred, np.ndarray)
        assert y_pred.dtype == np.int_
