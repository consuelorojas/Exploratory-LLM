import tensorflow as tf
import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






import numpy as np
from digits_classifier.interfaces import IClassifyDigits
from tensorflow.keras.datasets import mnist
import pytest

class TestDigitClassification:
    @pytest.fixture
    def classifier(self):
        from digits_classifier.classifier import ClassifyDigits
        return ClassifyDigits()

    @pytest.mark.parametrize("num_samples", [100, 500, 1000])
    def test_classification_accuracy(self, num_samples: int, classifier: IClassifyDigits):
        # Load MNIST dataset for testing
        (x_train, y_train), (_, _) = mnist.load_data()
        
        # Select a random subset of the training data
        indices = np.random.choice(len(x_train), size=num_samples)
        x_test = x_train[indices]
        y_test = y_train[indices]

        # Preprocess images to match input format expected by classifier
        x_test = x_test.reshape(-1, 28, 28) / 255.0

        # Classify digits using the model under test
        predictions = np.array([int(np.argmax(prediction)) for prediction in 
                                classifier(x_test)])

        # Calculate accuracy of classification results
        accuracy = np.mean(predictions == y_test)

        assert accuracy >= 0.95, f"Classification accuracy {accuracy:.2f} is below threshold (95%)"
