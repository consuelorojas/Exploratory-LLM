import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






import numpy as np
from digits_classifier.interfaces import IClassifyDigits
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
import pytest

class TestDigitClassification:
    @pytest.fixture
    def classifier(self):
        return ClassifyDigits()

    @pytest.mark.parametrize("num_samples", [100, 500, 1000])
    def test_classification_accuracy(self, num_samples: int, classifier: IClassifyDigits) -> None:
        # Load MNIST dataset
        (x_train, y_train), (_, _) = mnist.load_data()
        
        # Select a random subset of the training data for testing
        indices = np.random.choice(len(x_train), size=num_samples)
        x_test = x_train[indices]
        y_test = y_train[indices]

        # Preprocess images to match model input shape and type
        x_test = x_test.reshape(-1, 28 * 28) / 255.0

        # Classify digits using the classifier under test
        predictions = np.array([int(np.argmax(classifier(images=x_test[i].reshape(1, -1)))) for i in range(num_samples)])

        # Calculate accuracy of classifications
        accuracy = accuracy_score(y_true=y_test[:num_samples], y_pred=predictions)

        # Assert that the classification accuracy is at least 95%
        assert accuracy >= 0.95

    def test_invalid_input_shape(self, classifier: IClassifyDigits) -> None:
        with pytest.raises(ValueError):
            classifier(images=np.array([1]))

