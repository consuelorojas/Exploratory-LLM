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

        # Preprocess images to match model input shape and type
        x_test_images: np.ndarray = np.array([np.array(Image.fromarray(image).resize((28, 28))) for image in x_test])
        
        classifier: IClassifyDigits = ClassifyDigits()
        
        predictions: np.ndarray[np.int_] = classifier(x_test_images)

        # Calculate accuracy of the model
        accuracy: float = accuracy_score(y_test[:len(predictions)], predictions)
        
        assert round(accuracy, 2) >= 0.95

    def test_model_output_type(self):
        (x_train, y_train), (_, _) = mnist.load_data()
        x_test_images: np.ndarray = np.array([np.array(Image.fromarray(image).resize((28, 28))) for image in x_train[:1]])
        
        classifier: IClassifyDigits = ClassifyDigits()

        predictions: np.ndarray[np.int_] = classifier(x_test_images)

        assert isinstance(predictions[0], int)
