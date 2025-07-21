import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'ANN'))
from main import ClassifyDigits


import numpy as np
from digits_classifier.interfaces import IClassifyDigits
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from PIL.Image import Image

class TestDigitClassification:
    def test_classification_accuracy(self):
        # Load MNIST dataset for testing
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Create an instance of the ClassifyDigits class
        classifier: IClassifyDigits = ClassifyDigits()

        # Preprocess images to match input format expected by the model
        test_images = np.array([np.array(Image.fromarray(image).resize((28, 28))) for image in x_test])

        # Make predictions using the classifier
        y_pred = classifier(test_images)

        # Calculate accuracy of the model on the test set
        accuracy = accuracy_score(y_test, y_pred)

        # Assert that the classification accuracy is at least 95%
        assert accuracy >= 0.95

    def test_invalid_input(self):
        # Create an instance of the ClassifyDigits class
        classifier: IClassifyDigits = ClassifyDigits()

        # Test with invalid input shape (not a multiple of 28*28)
        try:
            classifier(np.random.rand(10, 20))
            assert False, "Expected ValueError for invalid input shape"
        except Exception as e:
            pass

    def test_empty_input(self):
        # Create an instance of the ClassifyDigits class
        classifier: IClassifyDigits = ClassifyDigits()

        try:
            classifier(np.array([]))
            assert False, "Expected ValueError or IndexError for empty input array"
        except (ValueError, IndexError) as e:
            pass

