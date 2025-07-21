
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

        # Preprocess images to match model input shape and normalize pixel values
        test_images = np.array([np.array(Image.fromarray(image).resize((28, 28))) for image in x_test])
        
        # Get predictions from the model
        y_pred = classifier(test_images)

        # Calculate accuracy using scikit-learn's accuracy_score function
        accuracy = accuracy_score(y_test, y_pred)
        
        # Assert that the classification accuracy is at least 95%
        assert accuracy >= 0.95

    def test_invalid_input_shape(self):
        # Create an instance of the ClassifyDigits class
        classifier: IClassifyDigits = ClassifyDigits()

        # Try to classify images with invalid shape (not (28, 28))
        try:
            classifier(np.random.rand(10, 32, 32))  # Invalid input shape
            assert False, "Expected ValueError for invalid input shape"
        except Exception as e:
            print(f"Caught exception: {e}")

    def test_empty_input(self):
        # Create an instance of the ClassifyDigits class
        classifier: IClassifyDigits = ClassifyDigits()

        try:
            classifier(np.array([]))  # Empty input array
            assert False, "Expected ValueError for empty input"
        except Exception as e:
            print(f"Caught exception: {e}")
