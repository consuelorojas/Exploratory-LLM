import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'ANN'))
from main import ClassifyDigits


import numpy as np
from digits_classifier.interfaces import IClassifyDigits
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
from PIL.Image import Image

def test_classification_accuracy():
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

def test_classification_accuracy_with_random_data():
    np.random.seed(42)
    random_images = np.random.randint(low=0, high=256, size=(100, 28, 28), dtype=np.uint8)

    # Create an instance of the ClassifyDigits class
    classifier: IClassifyDigits = ClassifyDigits()

    # Make predictions on test data
    y_pred = classifier(random_images / 255.0)

    # Assert that all predicted values are within valid range (i.e., between 0 and 9)
    assert np.all(y_pred >= 0) and np.all(y_pred <= 9)
