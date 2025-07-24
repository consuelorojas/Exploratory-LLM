import tensorflow as tf
import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






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

    def test_input_shape(self):
        # Create an instance of the ClassifyDigits class
        classifier: IClassifyDigits = ClassifyDigits()

        # Test with a single image
        img = np.random.rand(28, 28)
        result = classifier(np.array([img]))
        assert len(result) == 1

        # Test with multiple images
        imgs = [np.random.rand(28, 28) for _ in range(10)]
        results = classifier(np.array(imgs))
        assert len(results) == 10

    def test_output_type(self):
        # Create an instance of the ClassifyDigits class
        classifier: IClassifyDigits = ClassifyDigits()

        # Test with a single image
        img = np.random.rand(28, 28)
        result = classifier(np.array([img]))
        assert isinstance(result[0], int)

    def test_invalid_input(self):
        # Create an instance of the ClassifyDigits class
        classifier: IClassifyDigits = ClassifyDigits()

        # Test with invalid input shape (not a multiple of 784)
        img = np.random.rand(10, 28 * 29)  # Invalid shape
        try:
            result = classifier(img)
            assert False, "Expected ValueError"
        except Exception as e:
            assert isinstance(e, TypeError), f"Unexpected exception type: {type(e)}"

