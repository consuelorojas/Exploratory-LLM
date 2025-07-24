import tensorflow as tf
import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')





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

    @pytest.mark.parametrize("num_samples", [100, 500])
    def test_classification_accuracy(self, num_samples: int, classifier: IClassifyDigits) -> None:
        # Load MNIST dataset
        (x_train, y_train), (_, _) = mnist.load_data()
        
        # Select a subset of the training data for testing
        indices = np.random.choice(len(x_train), size=num_samples)
        x_test = x_train[indices]
        y_test = y_train[indices]

        # Preprocess images to match input format expected by classifier
        preprocessed_images = []
        for image in x_test:
            pil_image: Image = PIL.Image.fromarray(image, mode='L')
            resized_pil_image: Image = pil_image.resize((28, 28))
            numpy_array = np.array(resized_pil_image)
            preprocessed_images.append(numpy_array)

        # Classify digits
        predictions = classifier(np.array(preprocessed_images))

        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)

        assert accuracy >= 0.95

    def test_invalid_input(self, classifier: IClassifyDigits) -> None:
        with pytest.raises(ValueError):
            classifier(np.random.rand(1, 3))  # Invalid input shape (should be (n_samples, 28*28))

